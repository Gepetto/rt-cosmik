#!/usr/bin/env python3
"""Run one arm of the paper comparison over many trials and score each result.

Writes one summary row per trial to a CSV, and the usual joint_angles.csv /
markers.csv per run. Resumable: a trial already present in the summary is
skipped, so a long sweep can be interrupted and restarted, and a crash on one
trial costs only that trial.

    python3 scripts/python/paper/sweep.py --arm mmpose --cameras 0 2 4 6 \
        --dataset /root/workspace/COMFI --summary results/mmpose_4cam.csv

Only the scored DoF go into the summary means: with marker_set = "parity" the
7 locked DoF are excluded, exactly as compare_to_mocap.py excludes them, so a
mean here and a mean there are the same number.
"""
import argparse
import csv
import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s", force=True)
LOGGER = logging.getLogger("sweep")
LOGGER.setLevel(logging.INFO)

FIELDS = ["arm", "participant", "task", "cameras", "n_horizon", "frames",
          "lag", "sync_r", "joint_rmse_mean", "joint_rmse_median",
          "scored_dof", "freeflyer_mm", "marker_mm", "root_orientation_deg",
          "ik_ms_median", "ik_ms_p95", "fps", "status"]


def _load_eval():
    """Import compare_to_mocap.py, which is a script rather than a module."""
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(run_dir, reference_dir, ev):
    """Compare one finished run against mocap, returning summary numbers."""
    run = ev.load_run(str(run_dir))
    reference = ev.load_run(str(reference_dir))
    lag, correlation = ev.estimate_lag(run, reference)
    aligned_run, aligned_reference = ev.apply_lag(run, reference, lag)
    joints, freeflyer = ev.compare_joint_angles(aligned_run, aligned_reference)
    markers = ev.compare_markers(aligned_run, aligned_reference)

    angular = [r for r in joints if r["unit"] == "deg" and not r.get("locked")]
    linear = [r for r in joints if r["unit"] == "m"]
    rmse = [r["rmse"] for r in angular]
    return {
        "lag": lag,
        "sync_r": round(float(correlation), 4),
        "joint_rmse_mean": round(float(np.mean(rmse)), 3) if rmse else "",
        "joint_rmse_median": round(float(np.median(rmse)), 3) if rmse else "",
        "scored_dof": len(angular),
        "freeflyer_mm": (round(float(np.mean([r["rmse"] for r in linear])) * 1000, 2)
                         if linear else ""),
        "marker_mm": (round(float(np.mean([r["mean"] for r in markers])) * 1000, 2)
                      if markers else ""),
        "root_orientation_deg": round(float(freeflyer.get("orientation_deg", np.nan)), 3),
    }


def run_mmpose(dataset, participant, task, cameras, out_dir, settings,
               depth_aware=False):
    """One mmpose/LSTM trial. Returns (frames, ik_ms, seconds)."""
    from collections import OrderedDict
    from rtcosmik.paper.mmpose_baseline import build_source
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    source, meta = build_source(dataset, participant, task, cameras, settings,
                                depth_aware=depth_aware)
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(out_dir),
                     markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))

    ik_ms, rows = [], 0
    started = time.perf_counter()
    for frame, mks in source:
        t0 = time.perf_counter()
        q = solver.solve(mks)
        if rows:                      # the first call also builds the model
            ik_ms.append((time.perf_counter() - t0) * 1e3)
        markers = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            position = mks[name]
            markers[f"{name}_x"] = float(position[0])
            markers[f"{name}_y"] = float(position[1])
            markers[f"{name}_z"] = float(position[2])
        saver.save_markers(markers)
        saver.save_joint_angles(
            OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
        rows += 1
    saver.close()
    return rows, np.asarray(ik_ms), time.perf_counter() - started



_ESTIMATORS = {}


def _nlf_estimator(mtxs, num_cameras, settings):
    """One NLFEstimator per (camera count, intrinsics), reused across trials.

    Building it loads a YOLO engine and the NLF torchscript, tens of seconds, so
    it must not happen per trial. The intrinsics are baked in, so participants
    with different calibrations get their own.

    The torch backend flags must match run_pipeline's exactly. They change the
    numerics of NLF inference, not just its speed: without them this path and the
    real pipeline disagreed by up to 2.5 mm per marker and 1.8e-2 rad per joint
    on the same trial -- small, but a systematic offset rather than noise, and
    enough to make the sweep measure something other than the shipped pipeline.
    """
    import torch

    from rtcosmik.nlf.nlf import NLFEstimator
    from rtcosmik.model_weights import resolve_detector_engine

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    key = (num_cameras, tuple(np.asarray(m).round(4).tobytes() for m in mtxs))
    if key not in _ESTIMATORS:
        _ESTIMATORS.clear()          # only ever keep one on the GPU
        _ESTIMATORS[key] = NLFEstimator(
            yolo_path=resolve_detector_engine(settings.yolo_path, num_cameras),
            nlf_path=settings.nlf_path, cano_path=settings.cano_path,
            image_size=(settings.width, settings.height), cam_Ks=mtxs,
            indices=settings.nlf_indices, conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz, device=settings.device)
    return _ESTIMATORS[key]


def run_nlf(dataset, participant, task, cameras, out_dir, settings,
            depth_aware=False):
    """One NLF trial, mirroring run_pipeline's offline path. Returns (frames, ik_ms, seconds)."""
    from collections import OrderedDict, deque
    import yaml
    from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                           load_world_transformation)
    from rtcosmik.filtering.iir import IIR
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    mtxs, dists, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = load_world_transformation(cam_dir, cameras[0])

    paths = [root / "videos" / participant / task / f"camera_{c}.mp4" for c in cameras]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing video(s): {missing}")

    source = OfflineVideoSource(paths=[str(p) for p in paths],
                                size_wh=(settings.width, settings.height), loop=False)
    estimator = _nlf_estimator(mtxs, len(cameras), settings)
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))

    channels = 3 * len(settings.marker_names)
    iir = IIR(num_channel=channels, sampling_frequency=settings.fs)
    iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                   filter_type=settings.filter_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(out_dir),
                     markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))

    buffer = deque(maxlen=settings.N)
    ik_ms, rows, read = [], 0, 0
    started = time.perf_counter()
    while True:
        frames = source.read()
        if frames is None:
            break
        read += 1
        nlf_out, _, _, _ = estimator.estimate_from_frames(frames)
        p3d = reconstruct_3d(extract_views(nlf_out, len(cameras)), projections)
        if len(p3d) == 0:
            continue
        p3d = np.asarray(p3d) @ np.asarray(world_R).T + np.asarray(world_T)

        if not buffer:
            for _ in range(settings.N):
                buffer.append(p3d)
        else:
            buffer.append(p3d)
        block = np.asarray(buffer).reshape(settings.N, channels)
        markers_xyz = iir.filter(block).reshape(
            settings.N, len(settings.marker_names), 3)[-1]
        mks = dict(zip(settings.marker_names, markers_xyz))

        t0 = time.perf_counter()
        q = solver.solve(mks)
        if rows:
            ik_ms.append((time.perf_counter() - t0) * 1e3)

        markers = OrderedDict([("Frame_0", read - 1)])
        for name in settings.marker_names:
            position = mks[name]
            markers[f"{name}_x"] = float(position[0])
            markers[f"{name}_y"] = float(position[1])
            markers[f"{name}_z"] = float(position[2])
        saver.save_markers(markers)
        saver.save_joint_angles(
            OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
        rows += 1
    saver.close()
    return rows, np.asarray(ik_ms), time.perf_counter() - started


ARMS = {"mmpose": run_mmpose, "nlf": run_nlf}


def discover(dataset, participants, tasks, arm="mmpose"):
    """Trials with a mocap reference and whatever the arm needs to run.

    The mmpose arm reads precomputed 2D keypoints; the NLF arm reads the videos.
    Listing by mmpose output for both keeps the two arms on the same trial set,
    which is what makes their summaries comparable.
    """
    root = Path(dataset)
    found = []
    for participant in sorted(p.name for p in (root / "mmpose" / "output").iterdir()
                              if p.is_dir()):
        if participants and participant not in participants:
            continue
        for task in sorted(t.name for t in
                           (root / "mmpose" / "output" / participant).iterdir()
                           if t.is_dir()):
            if tasks and task not in tasks:
                continue
            if (root / "mocap" / "aligned" / participant / task).is_dir():
                found.append((participant, task))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["mmpose", "nlf"], default="mmpose")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", required=True)
    ap.add_argument("--summary", required=True, help="CSV to append results to")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--tag", default=None,
                    help="Run directory name; defaults to <n>cam_<arm>")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = _load_eval()

    tag = args.tag or f"{len(args.cameras)}cam_{args.arm}"
    cameras_key = "-".join(str(c) for c in args.cameras)
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if summary.exists():
        with open(summary) as handle:
            for row in csv.DictReader(handle):
                done.add((row["arm"], row["participant"], row["task"],
                          row["cameras"], row["n_horizon"]))
    fresh = not summary.exists()
    out = open(summary, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()

    trials = discover(args.dataset, args.participants, args.tasks, args.arm)
    LOGGER.info(f"{len(trials)} trials; {len(done)} already summarised; "
                f"cameras {cameras_key}; N={settings.N}; "
                f"marker_set={settings.marker_set}")

    for index, (participant, task) in enumerate(trials, 1):
        key = (args.arm, participant, task, cameras_key, str(settings.N))
        if key in done:
            continue
        out_dir = Path(settings.output_dir) / participant / task / tag
        row = dict.fromkeys(FIELDS, "")
        row.update(arm=args.arm, participant=participant, task=task,
                   cameras=cameras_key, n_horizon=settings.N)
        try:
            frames, ik_ms, seconds = ARMS[args.arm](
                args.dataset, participant, task, args.cameras, out_dir, settings)
            row.update(frames=frames,
                       ik_ms_median=round(float(np.median(ik_ms)), 2) if len(ik_ms) else "",
                       ik_ms_p95=round(float(np.percentile(ik_ms, 95)), 2) if len(ik_ms) else "",
                       fps=round(frames / seconds, 1))
            row.update(score(out_dir,
                             Path(args.dataset) / "mocap" / "aligned" / participant / task,
                             ev))
            row["status"] = "ok"
            LOGGER.info(f"[{index}/{len(trials)}] {participant}/{task}: "
                        f"{row['joint_rmse_mean']} deg over {row['scored_dof']} DoF, "
                        f"r={row['sync_r']}, {row['fps']} fps")
        except Exception as exc:
            row["status"] = f"{type(exc).__name__}: {exc}"[:200]
            LOGGER.warning(f"[{index}/{len(trials)}] {participant}/{task} FAILED: "
                           f"{row['status']}")
            LOGGER.debug(traceback.format_exc())
        writer.writerow(row)
        out.flush()
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
