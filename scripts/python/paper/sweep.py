#!/usr/bin/env python3
"""Run one arm of the paper comparison over many trials and score each result.

Writes one summary row per trial to a CSV, and the usual joint_angles.csv /
markers.csv per run. Resumable: a trial already present in the summary is
skipped, so a long sweep can be interrupted and restarted, and a crash on one
trial costs only that trial. A pass with another NLF backend or detection mode
needs its own --tag and --summary: the default tag carries ``_trt`` / ``_tiled``,
and a run folder made with another estimator is refused.

    python3 scripts/python/paper/sweep.py --arm mmpose --cameras 0 2 4 6 \
        --dataset /root/workspace/COMFI --summary results/mmpose_4cam.csv

Only the scored DoF go into the summary means: with marker_set = "parity" the
7 locked DoF are excluded, exactly as compare_to_mocap.py excludes them, so a
mean here and a mean there are the same number.
"""
import argparse
import csv
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

from rtcosmik.paper.scoring import ROW_FIELDS

FIELDS = (["arm", "participant", "task", "cameras", "n_horizon", "frames"] + ROW_FIELDS
          + ["ik_ms_median", "ik_ms_p95", "fps", "status"])


def _start(out_dir, settings, name, dataset, participant, task, cameras, landmark_map=None,
           **extra):
    from rtcosmik.paper import require_paper_evaluation
    from rtcosmik.provenance import start_run
    require_paper_evaluation(settings)
    start_run(out_dir, settings, dict(name=name, tag=Path(out_dir).name, cameras=list(cameras),
                                      participant=participant, task=task, dataset=str(dataset),
                                      **extra),
              landmark_map=landmark_map)


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
    _start(out_dir, settings, "mmpose", dataset, participant, task, cameras, depth_aware=depth_aware)
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


def set_torch_flags():
    """run_pipeline's torch backend flags; they change NLF's numerics, not only its speed."""
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _nlf_estimator(mtxs, num_cameras, settings, indices=None):
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
    from rtcosmik.nlf.nlf import NLFEstimator
    from rtcosmik.model_weights import estimator_arguments

    set_torch_flags()
    if indices is None:
        indices = settings.nlf_indices
    key = (settings.nlf_backend, settings.nlf_engine_path, settings.detection_mode,
           tuple(settings.yolo_tiled_imgsz), num_cameras, tuple(indices),
           tuple(np.asarray(m).round(4).tobytes() for m in mtxs))
    if key not in _ESTIMATORS:
        _ESTIMATORS.clear()          # only ever keep one on the GPU
        _ESTIMATORS[key] = NLFEstimator(
            **estimator_arguments(settings, num_cameras), cano_path=settings.cano_path,
            image_size=(settings.width, settings.height), cam_Ks=mtxs,
            indices=indices, conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz, device=settings.device)
    return _ESTIMATORS[key]


def run_nlf(dataset, participant, task, cameras, out_dir, settings,
            depth_aware=False, reconstruction="fuse3d", views_cache=None):
    """One NLF trial, mirroring run_pipeline's offline path. Returns (frames, ik_ms, seconds).

    ``reconstruction`` selects what is done with NLF's output. "fuse3d" is the
    shipped path: inverse-variance fusion of the metric 3D pose each view
    regresses. "tri2d" instead triangulates NLF's own 2D keypoints with the same
    weighted DLT the mmpose arm uses, which holds the reconstruction fixed and
    leaves the detector as the only difference between the two arms.

    With ``views_cache``, every frame's per-camera NLF output (2D keypoints,
    metric 3D pose, uncertainty) is also saved to ``<views_cache>/<participant>/
    <task>.npz``, so studies that only change what happens after NLF -- IK type,
    horizon, filter -- can replay it without the GPU. Saving happens after the
    timed loop.
    """
    from collections import OrderedDict
    import yaml
    from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                           load_world_transformation)
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik import landmark_maps
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver
    from rtcosmik.triangulation.triangulation import (reconstruct_3d,
                                                      triangulate_points)
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    landmark_map = landmark_maps.select(settings, participant)
    landmark_maps.require_out_of_fold(landmark_map, participant, settings)
    settings = landmark_maps.apply(landmark_map, settings)
    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    # No camera-id correction here. 3361's swap is a defect in COMFI's mmpose 2D
    # export, not in the rig: the videos this arm reads are labelled correctly,
    # and applying the mmpose correction to them breaks the reconstruction
    # (12.23 -> 14.78 deg, 56 -> 595 mm on 3361/Lifting). Verified by running
    # both ways rather than assumed, after assuming wrongly once.
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

    marker_filter = MarkerFilter(len(settings.marker_names), settings)

    _start(out_dir, settings, "nlf", dataset, participant, task, cameras,
           landmark_map.identity(participant), reconstruction=reconstruction)
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(out_dir),
                     markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))

    ik_ms, rows, read = [], 0, 0
    recorder = None
    if views_cache is not None:
        from rtcosmik.paper.nlf_views import ViewsRecorder
        from rtcosmik.provenance import estimator_identity
        recorder = ViewsRecorder(cameras, len(settings.marker_names), landmark_map,
                                 estimator_identity(settings, len(cameras)))
    started = time.perf_counter()
    try:
        while True:
            frames = source.read()
            if frames is None:
                break
            read += 1
            nlf_out, _, _, _ = estimator.estimate_from_frames(frames)
            views = extract_views(nlf_out, len(cameras))
            if recorder is not None:
                recorder.append(views)
            if reconstruction == "tri2d":
                if any(k is None for k in views.keypoints) or len(cameras) < 2:
                    continue
                p3d = triangulate_points(views.keypoints, mtxs, dists, projections,
                                         uncertainties=views.uncertainties)
            else:
                p3d = reconstruct_3d(views, projections)
            if len(p3d) == 0:
                continue
            p3d = np.asarray(p3d) @ np.asarray(world_R).T + np.asarray(world_T)

            mks = dict(zip(settings.marker_names, marker_filter(p3d)))

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
    finally:
        # ffmpeg keeps running if nobody tears it down: the reader
        # thread blocks on a full queue rather than reaching EOF, so
        # the decoder sits holding GPU memory for the life of the
        # process. Four cameras a trial adds up fast.
        source.release()
    seconds = time.perf_counter() - started
    if recorder is not None:
        recorder.save(Path(views_cache) / participant / f"{task}.npz")
    return rows, np.asarray(ik_ms), seconds


def run_mocap(dataset, participant, task, cameras, out_dir, settings,
              depth_aware=False):
    """Drive the same model and IK from mocap markers. Returns (frames, ik_ms, seconds).

    ``cameras`` is ignored -- mocap has no cameras -- but kept in the signature so
    the sweep can treat this like any other arm.
    """
    from collections import OrderedDict
    import yaml
    from rtcosmik.paper.mocap_reference import MocapMarkerSource
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    source = MocapMarkerSource(root / "mocap" / "aligned" / participant / task,
                               settings.marker_names)
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    _start(out_dir, settings, "mocap", dataset, participant, task, cameras)
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(out_dir),
                     markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))

    ik_ms, rows = [], 0
    started = time.perf_counter()
    for frame, mks in source:
        t0 = time.perf_counter()
        q = solver.solve(mks)
        if rows:
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


def run_nlf2d(dataset, participant, task, cameras, out_dir, settings,
              depth_aware=False):
    """NLF's 2D keypoints through the same weighted DLT the mmpose arm uses."""
    return run_nlf(dataset, participant, task, cameras, out_dir, settings,
                   depth_aware=depth_aware, reconstruction="tri2d")


def run_fastsam(dataset, participant, task, cameras, out_dir, settings,
                depth_aware=False):
    """One FastSAM-3D trial. Returns (frames, ik_ms, seconds).

    COMFI exports this modality as metric 3D markers in the reference camera's
    frame, so the arm reads a file rather than running a network: the timings it
    reports are the IK's alone and are not comparable with the end-to-end fps of
    the arms that include a detector.
    """
    from collections import OrderedDict
    from rtcosmik import landmark_maps
    from rtcosmik.paper.fastsam_source import build_source
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    landmark_map = landmark_maps.submitted(settings, "fastsam")     # the COMFI exports
    landmark_maps.require_out_of_fold(landmark_map, participant, settings)
    source, meta = build_source(dataset, participant, task, cameras, settings)
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    _start(out_dir, settings, "fastsam", dataset, participant, task, cameras,
           landmark_map.identity(participant))
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


NLF_ARMS = ("nlf", "nlf2d")
_ESTIMATOR_KEYS = ("nlf_backend", "detection_mode", "tiled_imgsz")


def _estimator(settings, cameras):
    from rtcosmik.provenance import estimator_identity
    identity = estimator_identity(settings, len(cameras))
    return {k: identity.get(k) for k in _ESTIMATOR_KEYS}


def default_tag(arm, cameras, settings):
    """``<n>cam_<arm>``, with ``_trt`` and ``_tiled`` when an NLF arm runs them."""
    tag = f"{len(cameras)}cam_{arm}"
    if arm in NLF_ARMS:
        estimator = _estimator(settings, cameras)
        tag += "_trt" * (estimator["nlf_backend"] == "tensorrt")
        tag += "_tiled" * (estimator["detection_mode"] == "tiled")
    return tag


def require_same_estimator(out_dir, cameras, settings):
    """Refuse a run folder made with another NLF backend or detection mode."""
    import json
    path = Path(out_dir) / "run_info.json"
    if not path.exists():
        return
    made = {"nlf_backend": "torchscript", "detection_mode": "per_view",
            **json.loads(path.read_text()).get("estimator", {})}
    made = {k: made.get(k) for k in _ESTIMATOR_KEYS}
    now = _estimator(settings, cameras)
    if made != now:
        raise RuntimeError(f"{out_dir} was run with {made}, the settings now give {now}; "
                           "pass another --tag and --summary")


ARMS = {"mmpose": run_mmpose, "nlf": run_nlf, "nlf2d": run_nlf2d,
        "fastsam": run_fastsam, "mocap": run_mocap}


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
    ap.add_argument("--arm",
                    choices=["mmpose", "nlf", "nlf2d", "fastsam", "mocap"],
                    default="mmpose")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", required=True)
    ap.add_argument("--summary", required=True, help="CSV to append results to")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--tag", default=None,
                    help="Run directory name; defaults to <n>cam_<arm>[_trt][_tiled]")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="where run folders go; defaults to settings.output_dir")
    ap.add_argument("--views-cache", type=Path, default=None,
                    help="nlf arm only: also save per-frame NLF views here, for replay studies")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    from rtcosmik.provenance import require_clean
    require_clean()
    if args.arm == "mocap":
        # Mocap has no facial landmarks, so it runs a 32-marker set of its own.
        # The OCP for it must already exist; the fingerprint check refuses a
        # mismatched one rather than silently reusing the parity build.
        from rtcosmik.marker_sets import apply_marker_set
        apply_marker_set(settings, "mocap")
    from rtcosmik.paper.scoring import score_run

    tag = args.tag or default_tag(args.arm, args.cameras, settings)
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
    runs_root = args.output_dir or Path(settings.output_dir)
    if args.arm in NLF_ARMS:
        for participant, task in trials:
            require_same_estimator(runs_root / participant / task / tag, args.cameras, settings)
    LOGGER.info(f"{len(trials)} trials; {len(done)} already summarised; "
                f"cameras {cameras_key}; N={settings.N}; "
                f"marker_set={settings.marker_set}")

    for index, (participant, task) in enumerate(trials, 1):
        key = (args.arm, participant, task, cameras_key, str(settings.N))
        if key in done:
            continue
        out_dir = runs_root / participant / task / tag
        row = dict.fromkeys(FIELDS, "")
        row.update(arm=args.arm, participant=participant, task=task,
                   cameras=cameras_key, n_horizon=settings.N)
        try:
            extra = {"views_cache": args.views_cache} if args.views_cache and args.arm == "nlf" else {}
            frames, ik_ms, seconds = ARMS[args.arm](
                args.dataset, participant, task, args.cameras, out_dir, settings, **extra)
            row.update(frames=frames,
                       ik_ms_median=round(float(np.median(ik_ms)), 2) if len(ik_ms) else "",
                       ik_ms_p95=round(float(np.percentile(ik_ms, 95)), 2) if len(ik_ms) else "",
                       fps=round(frames / seconds, 1))
            row.update(score_run(out_dir, args.dataset, participant, task, settings))
            row["status"] = "ok"
            LOGGER.info(f"[{index}/{len(trials)}] {participant}/{task}: "
                        f"{row['aligned_rmse_mean']} deg over {row['scored_dof']} DoF, "
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
