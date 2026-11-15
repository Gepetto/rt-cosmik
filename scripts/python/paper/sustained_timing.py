#!/usr/bin/env python3
"""Sustained timing of the pipeline on recorded video, and what COMFI's timestamps say.

Stages, each writing under ``--out``:

``replay``      the trials of ``SUBSET`` through the whole pipeline (decode,
                detection, NLF, fusion, filter, IK, output), frames released at
                the camera rate, one pass per inference configuration.
``opencap``     the same on one OpenCap subject (``--dataset`` is then
                LabValidation_withVideos): four cameras at 60 Hz.
``ik``          the recorded NLF output of the same trials through the
                moving-horizon and the per-frame IK, one after the other.
``timestamps``  dropped frames and inter-camera spread of every COMFI trial of
                the paper, from the per-camera timestamp files.
``report``      ``timing.md`` from the three.

Run on an idle machine. Procedure: docs/research/timing.md.

    python3 scripts/python/paper/sustained_timing.py replay --dataset /root/workspace/COMFI --out results/timing
"""
import argparse
import csv
import json
import logging
import multiprocessing
import os
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
#: The trials timed: one participant, the six tasks, about six minutes of video.
SUBSET = tuple(("1012", task) for task in TASKS)
CAMERAS = (0, 2, 4, 6)
#: name -> (NLF backend, detection mode)
CONFIGURATIONS = {
    "torchscript": ("torchscript", "per_view"),
    "tensorrt": ("tensorrt", "per_view"),
    "tensorrt_tiled": ("tensorrt", "tiled"),
}
#: OpenCap LabValidation: one subject, every dynamic trial, four of its five cameras at 60 Hz.
OPENCAP_SUBJECT = "subject4"
OPENCAP_CAMERAS = (0, 1, 3, 4)
IK_TYPES = {"mhe": "Moving-horizon IK", "sbs": "Per-frame IK"}
STAGES = ("decode", "upload", "detection", "crop", "nlf", "fusion", "filter", "ik", "output")
WARMUP_FRAMES = 40           # calibration and first solves, left out of the statistics
#: Lag of the paper's marker filter, in frames (resubmission-findings.md, section 0d).
FILTER_LAG_FRAMES = 0.4


def _settings():
    from rtcosmik.config_loader import settings
    from rtcosmik.paper import require_paper_evaluation
    require_paper_evaluation(settings)
    return settings


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def timed_trial(source, estimator, projections, world, settings, solve_first, solver, run_dir):
    """One trial through the pipeline, frames released at the camera rate; per-frame durations (ms)."""
    import torch
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.saver.csv_saver import CSVSaver
    from rtcosmik.triangulation.triangulation import reconstruct_3d

    world_R, world_T = world
    period = 1.0 / settings.fs
    estimator.reset_tracking()
    estimator.sync_timings = True
    marker_filter = MarkerFilter(len(settings.marker_names), settings)
    saver = CSVSaver(str(run_dir), markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    rows, frame, started = [], 0, time.perf_counter()
    while True:
        t0 = time.perf_counter()
        frames = source.read()
        if frames is None:
            break
        t1 = time.perf_counter()
        time.sleep(max(0.0, started + frame * period - t1))       # the frame is not there earlier
        t2 = time.perf_counter()
        output, stage, _, _ = estimator.estimate_from_frames(frames)
        views = extract_views(output, len(projections))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        points = reconstruct_3d(views, projections, settings.fusion)
        if len(points) == 0:
            frame += 1
            continue
        points = np.asarray(points) @ world_R.T + world_T
        t4 = time.perf_counter()
        markers = dict(zip(settings.marker_names, marker_filter(points)))
        t5 = time.perf_counter()
        q = solver.solve(markers) if rows else solve_first(markers)
        t6 = time.perf_counter()
        row = OrderedDict([("Frame_0", frame)])
        for marker in settings.marker_names:
            row[f"{marker}_x"], row[f"{marker}_y"], row[f"{marker}_z"] = map(float, markers[marker])
        saver.save_markers(row)
        saver.save_joint_angles(OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
        t7 = time.perf_counter()
        named = [stage["h2d+pre_ms"], stage["yolo_ms"], stage.get("crop_ms", 0.0), stage["nlf_ms"]]
        rows.append([frame, 1e3 * (t1 - t0), *named, 1e3 * (t4 - t3), 1e3 * (t5 - t4),
                     1e3 * (t6 - t5), 1e3 * (t7 - t6), 1e3 * (t3 - t2) - sum(named), 1e3 * (t7 - t2)])
        frame += 1
    saver.close()
    return rows


COLUMNS = ["frame", *STAGES, "inference_other", "processing"]


def _describe(folder, settings, cameras, trials):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pass.json").write_text(json.dumps(
        {"period_ms": 1e3 / settings.fs, "cameras": list(cameras), "trials": trials,
         "image": [settings.width, settings.height]}))


def replay_configuration(job):
    """One configuration over the COMFI subset."""
    import yaml
    import sweep
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    dataset, out, name = job
    base = _settings()
    base.nlf_backend, base.detection_mode = CONFIGURATIONS[name]
    cameras = list(CAMERAS)
    folder = out / "replay" / name
    _describe(folder, base, cameras, len(SUBSET))
    for participant, task in SUBSET:
        landmark_map = landmark_maps.select(base, participant)
        landmark_maps.require_out_of_fold(landmark_map, participant, base)
        settings = landmark_maps.apply(landmark_map, base)
        cam_dir = dataset / "cam_params" / participant
        mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
        world = [np.asarray(a) for a in load_world_transformation(cam_dir, cameras[0])]
        meta = yaml.safe_load((dataset / "metadata" / f"{participant}.yaml").read_text())
        source = OfflineVideoSource(
            paths=[str(dataset / "videos" / participant / task / f"camera_{c}.mp4") for c in cameras],
            size_wh=(settings.width, settings.height), loop=False)
        solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                             weight=meta["weight"], logger=logging.getLogger("solve"))
        rows = timed_trial(source, sweep._nlf_estimator(mtxs, len(cameras), settings), projections,
                           world, settings, solver.solve, solver, folder / f"{participant}_{task}")
        _write(folder / f"{participant}_{task}.csv", COLUMNS, rows)
    return name


def replay_opencap(job):
    """One configuration over the OpenCap subject: calibrated on the static clip, then every trial."""
    import sweep
    import opencap_eval
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.paper import opencap as oc
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    dataset, out, name = job
    base = opencap_eval._settings()
    base.nlf_backend, base.detection_mode = CONFIGURATIONS[name]
    cameras = list(OPENCAP_CAMERAS)
    subject_dir = dataset / OPENCAP_SUBJECT
    landmark_map = landmark_maps.select(base, OPENCAP_SUBJECT)
    landmark_maps.require_out_of_fold(landmark_map, OPENCAP_SUBJECT, base)
    settings = landmark_maps.apply(landmark_map, base)
    trials = oc.trials(subject_dir)
    folder = out / "opencap" / name
    _describe(folder, settings, cameras, len(trials))

    def setup(trial):
        cam_dir = opencap_eval._calibration(dataset, out, OPENCAP_SUBJECT, trial.session)
        mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
        world = [np.asarray(a) for a in load_world_transformation(cam_dir, cameras[0])]
        source = OfflineVideoSource(
            paths=[str(oc.video_path(subject_dir, trial.session, c, trial.name)) for c in cameras],
            size_wh=(settings.width, settings.height), loop=False)
        return source, sweep._nlf_estimator(mtxs, len(cameras), settings), projections, world

    static = oc.Trial(oc.STATIC, oc.session_of(subject_dir, oc.STATIC), None)
    source, estimator, projections, world = setup(static)
    estimator.reset_tracking()
    postures = []
    while (frames := source.read()) is not None:
        points = reconstruct_3d(extract_views(estimator.estimate_from_frames(frames)[0], len(cameras)),
                                projections, settings.fusion)
        if len(points):
            postures.append(np.asarray(points) @ world[0].T + world[1])
    solver = HumanSolver(settings, **opencap_eval._anthropometry(subject_dir),
                         logger=logging.getLogger("solve"))
    solver.calibrate(dict(zip(settings.marker_names, np.median(postures, axis=0))))
    for trial in trials:
        source, estimator, projections, world = setup(trial)
        rows = timed_trial(source, estimator, projections, world, settings, solver.start_trial,
                           solver, folder / trial.name)
        _write(folder / f"{trial.name}.csv", COLUMNS, rows)
    return name


def ik_cost(job):
    """Solve time (ms) per frame of one IK type on the subset's recorded NLF output."""
    import yaml
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.landmark_maps import checked_recording
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.pipeline.solver import HumanSolver

    dataset, views, out, ik_type = job
    settings = _settings()
    settings.ik_type = ik_type
    for participant, task in SUBSET:
        recording = views / participant / f"{task}.npz"
        checked_recording(recording, participant, settings)
        meta = yaml.safe_load((dataset / "metadata" / f"{participant}.yaml").read_text())
        solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                             weight=meta["weight"], logger=logging.getLogger("solve"))
        marker_filter = MarkerFilter(len(settings.marker_names), settings)
        rows = []
        for frame, points in replay_markers(recording, dataset / "cam_params" / participant, list(CAMERAS)):
            markers = dict(zip(settings.marker_names, marker_filter(points)))
            t0 = time.perf_counter()
            solver.solve(markers)
            rows.append([frame, 1e3 * (time.perf_counter() - t0)])
        _write(out / "ik" / ik_type / f"{participant}_{task}.csv", ["frame", "ik"], rows)
    return ik_type


def camera_times(trial_dir, cameras=CAMERAS):
    """(F, C) arrival times in seconds of a trial's frames, or None without timestamp files."""
    columns = []
    for camera in cameras:
        path = trial_dir / f"camera_{camera}_timestamps.csv"
        if not path.exists():
            return None
        with open(path) as handle:
            stamps = [row["timestamp"] for row in csv.DictReader(handle)]
        columns.append(np.array(stamps, dtype="datetime64[us]").astype(np.int64) * 1e-6)
    frames = min(len(c) for c in columns)
    return np.stack([c[:frames] for c in columns], axis=1)


def timestamps(dataset, out, period_s):
    """Timestamp report of every paper trial, one row each."""
    from rtcosmik.folds import PARTICIPANTS
    from rtcosmik.paper.timing import timestamp_report
    rows, missing = [], 0
    for participant in PARTICIPANTS:
        for task in TASKS:
            times = camera_times(dataset / "videos" / participant / task)
            if times is None:
                missing += 1
                continue
            report = timestamp_report(times, period_s)
            rows.append([participant, task, report["frames"], *report["missing_frames"],
                         *report["long_intervals"], report["spread_ms"]["median"],
                         report["spread_ms"]["p95"], report["spread_ms"]["max"],
                         *report["offset_to_first_ms"],
                         *[s["mean"] for s in report["interval_ms"]],
                         *[s["max"] for s in report["interval_ms"]]])
    cams = [str(c) for c in CAMERAS]
    _write(out / "timestamps.csv",
           ["participant", "task", "frames", *[f"missing_{c}" for c in cams],
            *[f"long_{c}" for c in cams], "spread_median_ms", "spread_p95_ms", "spread_max_ms",
            *[f"offset_{c}_ms" for c in cams], *[f"interval_mean_{c}_ms" for c in cams],
            *[f"interval_max_{c}_ms" for c in cams]], rows)
    return len(rows), missing


def machine():
    """What ran the passes, for the report."""
    import torch
    return {"host": platform.node(), "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count(), "gpu": torch.cuda.get_device_name(0),
            "load_average": os.getloadavg()[0], "date": time.strftime("%Y-%m-%d %H:%M")}


def _fresh(function, job):
    """Run in a new process, so nothing is shared between configurations."""
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(function, (job,))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["replay", "opencap", "ik", "timestamps", "report"])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=Path, help="recorded NLF output of the subset (stage ik)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "replay":
        (args.out / "machine.json").write_text(json.dumps(machine(), indent=1))
        for name in CONFIGURATIONS:
            print(f"replayed {_fresh(replay_configuration, (args.dataset, args.out, name))}", flush=True)
    elif args.stage == "opencap":
        for name in CONFIGURATIONS:
            print(f"replayed {_fresh(replay_opencap, (args.dataset, args.out, name))}", flush=True)
    elif args.stage == "ik":
        if args.views is None:
            ap.error("stage ik needs --views")
        for ik_type in IK_TYPES:
            print(f"timed {_fresh(ik_cost, (args.dataset, args.views, args.out, ik_type))}", flush=True)
    elif args.stage == "timestamps":
        trials, missing = timestamps(args.dataset, args.out, 1.0 / _settings().fs)
        print(f"{trials} trials with timestamps, {missing} without")
    else:
        import timing_report
        timing_report.write(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
