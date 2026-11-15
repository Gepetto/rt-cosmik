#!/usr/bin/env python3
"""Evaluate the paper pipeline on the OpenCap LabValidation dataset.

Stages, each resumable:

    reference   mocap markers through our model and IK (the reference runs)
    record      YOLO + NLF on the five videos of every trial, saved as views (GPU)
    replay      the arms of ``opencap.ARMS`` from the recorded views (CPU)
    report      scores and tables, with OpenCap's released angles beside ours

    python3 scripts/python/paper/opencap_eval.py reference \\
        --dataset /root/workspace/OpenCap/data/LabValidation_withVideos --out /root/workspace/oof/t16

The model is calibrated on the static trial, then every dynamic trial starts
from that calibration. Method and tables: docs/research/opencap-evaluation.md.
"""
import argparse
import json
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

from rtcosmik.paper import opencap as oc
from rtcosmik.paper.scoring import REFERENCE_TAG

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s",
                    force=True)
LOGGER = logging.getLogger("opencap")
LOGGER.setLevel(logging.INFO)


def _settings(marker_set=None):
    from rtcosmik.config_loader import settings
    from rtcosmik.paper import require_paper_evaluation
    require_paper_evaluation(settings)
    return oc.dataset_settings(settings, marker_set)


def _anthropometry(subject_dir):
    import yaml
    meta = yaml.safe_load((subject_dir / "sessionMetadata.yaml").read_text())
    return {"gender": meta["sex"][0], "height": meta["height_m"], "weight": meta["mass_kg"]}


def _calibration(dataset, out, subject, session):
    cam_dir = out / "calibration" / subject / f"Session{session}"
    if not (cam_dir / "intrinsics").is_dir():
        oc.write_calibration(dataset / subject, session, cam_dir)
    return cam_dir


def _done(run_dir):
    path = run_dir / "run_info.json"
    return path.exists() and "frames" in json.loads(path.read_text())


def _residual_mm(solver, q, mks, settings):
    """RMS distance between the model's markers at ``q`` and the measured ones."""
    import pinocchio as pin
    data = solver.model.createData()
    pin.framesForwardKinematics(solver.model, data, q)
    d = [data.oMf[solver.model.getFrameId(k)].translation - mks[k]
         for k in settings.keys_to_track_list]
    return float(np.sqrt(np.mean(np.sum(np.square(d), axis=1))) * 1000)


def _solve_trial(solver, frames, run_dir, settings):
    """Start the calibrated solver on a trial's first frame, solve the rest, write the CSVs."""
    from rtcosmik.provenance import update_run_info
    from rtcosmik.saver.csv_saver import CSVSaver
    saver = CSVSaver(str(run_dir), markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    ik_ms, start_residual = [], None
    for index, (frame, mks) in enumerate(frames):
        t0 = time.perf_counter()
        if index == 0:
            q = solver.start_trial(mks)
            start_residual = _residual_mm(solver, q, mks, settings)
        else:
            q = solver.solve(mks)
            ik_ms.append((time.perf_counter() - t0) * 1e3)
        markers = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            markers[f"{name}_x"], markers[f"{name}_y"], markers[f"{name}_z"] = map(float, mks[name])
        saver.save_markers(markers)
        saver.save_joint_angles(OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
    saver.close()
    update_run_info(run_dir, frames=len(frames), start_residual_mm=start_residual,
                    ik_ms_median=float(np.median(ik_ms)) if ik_ms else None)


def _median_frame(markers):
    return {name: np.nanmedian(xyz, axis=0) for name, xyz in markers.items()}


def reference_subject(job):
    """Reference runs of one subject: calibrated on the static mocap, one run per dynamic trial."""
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.provenance import start_run
    dataset, out, subject = job
    settings = _settings(oc.REFERENCE_SET)
    subject_dir = dataset / subject
    todo = [t for t in oc.trials(subject_dir)
            if not _done(out / "runs" / subject / t.name / REFERENCE_TAG)]
    if not todo:
        return subject, 0
    solver = HumanSolver(settings, **_anthropometry(subject_dir), logger=logging.getLogger("solve"))
    q = solver.calibrate(_median_frame(oc.reference_markers(oc.trc_path(subject_dir, oc.STATIC))))
    oc.require_standing(settings.joint_angles_names, q)
    for trial in todo:
        run_dir = out / "runs" / subject / trial.name / REFERENCE_TAG
        times = oc.frame_times(oc.video_pts(oc.video_path(subject_dir, trial.session, 0, trial.name)),
                               settings.fs)
        markers = oc.reference_markers(oc.trc_path(subject_dir, trial.name), at=times)
        frames = [(f, {n: markers[n][f] for n in settings.marker_names}) for f in range(len(times))]
        frames = [(f, m) for f, m in frames if all(np.isfinite(v).all() for v in m.values())]
        start_run(run_dir, settings, dict(name="mocap", tag=REFERENCE_TAG, participant=subject,
                                          task=trial.name, dataset=str(dataset), cameras=[],
                                          calibration=oc.STATIC), repo=REPO)
        _solve_trial(solver, frames, run_dir, settings)
        np.save(run_dir / "frame_times.npy", times)
    return subject, len(todo)


def record_subject(dataset, out, subject):
    """Per-view NLF output of the static and dynamic trials of one subject, five cameras."""
    import sweep
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.paper.nlf_views import ViewsRecorder
    from rtcosmik.provenance import estimator_identity
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    base = _settings()
    landmark_map = landmark_maps.select(base, subject)
    landmark_maps.require_out_of_fold(landmark_map, subject, base)
    settings = landmark_maps.apply(landmark_map, base)
    subject_dir = dataset / subject
    cameras = list(oc.CAMERAS)
    static = oc.Trial(oc.STATIC, oc.session_of(subject_dir, oc.STATIC), None)
    for trial in [static] + oc.trials(subject_dir):
        views_path = out / "views" / subject / f"{trial.name}.npz"
        if views_path.exists():
            continue
        cam_dir = _calibration(dataset, out, subject, trial.session)
        mtxs = load_camera_parameters(cam_dir, cameras)[0]
        estimator = sweep._nlf_estimator(mtxs, len(cameras), settings)
        estimator.reset_tracking()
        paths = [oc.video_path(subject_dir, trial.session, c, trial.name) for c in cameras]
        pts = oc.video_pts(paths[0])
        repeats = oc.leading_repeats(pts, settings.fs)    # the decoder fills the gap after frame 0
        recorder = ViewsRecorder(cameras, len(settings.marker_names), landmark_map,
                                 estimator_identity(settings, len(cameras)))
        source = OfflineVideoSource(paths=[str(p) for p in paths],
                                    size_wh=(settings.width, settings.height), loop=False)
        read = 0
        try:
            while True:
                frames = source.read()
                if frames is None:
                    break
                read += 1
                if read <= repeats:
                    continue
                nlf_out = estimator.estimate_from_frames(frames)[0]
                recorder.append(extract_views(nlf_out, len(cameras)))
        finally:
            source.release()
        if read - repeats != len(pts):
            raise RuntimeError(f"{subject}/{trial.name}: decoded {read} frames with {repeats} "
                               f"repeats, the clip holds {len(pts)}")
        recorder.save(views_path)
        LOGGER.info(f"recorded {subject}/{trial.name}: {len(pts)} frames")


def replay_arm(job):
    """One arm on one subject: calibrated on its own static landmarks, one run per dynamic trial."""
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.landmark_maps import checked_recording
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.provenance import recorded_estimator, start_run
    dataset, out, subject, tag = job
    settings = _settings()
    cameras = list(oc.ARMS[tag])
    subject_dir = dataset / subject
    views = lambda trial: out / "views" / subject / f"{trial}.npz"
    todo = [t for t in oc.trials(subject_dir) if not _done(out / "runs" / subject / t.name / tag)]
    if not todo:
        return subject, tag, 0
    static = replay_markers(views(oc.STATIC), _calibration(
        dataset, out, subject, oc.session_of(subject_dir, oc.STATIC)), cameras, fusion=settings.fusion)
    posture = np.median([p3d for _, p3d in static], axis=0)
    solver = HumanSolver(settings, **_anthropometry(subject_dir), logger=logging.getLogger("solve"))
    q = solver.calibrate(dict(zip(settings.marker_names, posture)))
    oc.require_standing(settings.joint_angles_names, q)
    for trial in todo:
        run_dir = out / "runs" / subject / trial.name / tag
        landmark_map = checked_recording(views(trial.name), subject, settings)
        start_run(run_dir, settings, dict(name="nlf", tag=tag, cameras=cameras, participant=subject,
                                          task=trial.name, dataset=str(dataset),
                                          views=str(views(trial.name)), reconstruction="fuse3d",
                                          calibration=oc.STATIC),
                  repo=REPO, landmark_map=landmark_map, estimator=recorded_estimator(views(trial.name)))
        marker_filter = MarkerFilter(len(settings.marker_names), settings)
        frames = [(frame, dict(zip(settings.marker_names, marker_filter(p3d))))
                  for frame, p3d in replay_markers(
                      views(trial.name), _calibration(dataset, out, subject, trial.session),
                      cameras, fusion=settings.fusion)]
        _solve_trial(solver, frames, run_dir, settings)
    return subject, tag, len(todo)


def _pool(function, jobs, workers):
    """One fresh process per job; the first runs alone, so the OCP is built once."""
    import multiprocessing
    if not jobs:
        return
    LOGGER.info(f"{function.__name__}: {len(jobs)} jobs")
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        LOGGER.info(f"  done {pool.apply(function, (jobs[0],))}")
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(function, jobs[1:]):
            LOGGER.info(f"  done {result}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["reference", "record", "replay", "report"])
    ap.add_argument("--dataset", type=Path, required=True, help="LabValidation_withVideos")
    ap.add_argument("--out", type=Path, required=True, help="views, run folders and tables go here")
    ap.add_argument("--subjects", nargs="*", default=list(oc.SUBJECTS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from rtcosmik.provenance import require_clean
    require_clean()
    if args.stage == "reference":
        _pool(reference_subject, [(args.dataset, args.out, s) for s in args.subjects], args.workers)
    elif args.stage == "record":
        for subject in args.subjects:
            record_subject(args.dataset, args.out, subject)
    elif args.stage == "replay":
        _pool(replay_arm, [(args.dataset, args.out, s, tag) for tag in oc.ARMS
                           for s in args.subjects], args.workers)
    else:
        import opencap_report
        opencap_report.write(args.dataset, args.out, args.subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
