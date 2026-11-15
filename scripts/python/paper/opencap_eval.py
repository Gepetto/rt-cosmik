#!/usr/bin/env python3
"""Evaluate the paper pipeline on the OpenCap LabValidation dataset.

Stages, each resumable:

    record      YOLO + NLF on the five videos of every trial, saved as views (GPU)
    sync        cameras synchronised on their landmarks, video clock set on the mocap's
    reference   mocap markers through our model and IK (the reference runs)
    replay      the arms of ``opencap.ARMS`` from the recorded views (CPU)
    report      scores and tables, with OpenCap's released angles beside ours

    python3 scripts/python/paper/opencap_eval.py record \\
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


def _views(out, subject, trial):
    """A trial's recorded views: synchronised ones for a dynamic trial, the clip's for the static."""
    return out / ("views" if trial == oc.STATIC else "synced") / subject / f"{trial}.npz"


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
        times = np.load(_views(out, subject, trial.name))["times"]
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
    """Per-view NLF output of the static and dynamic trials of one subject, five cameras.

    Row ``r`` of a recording holds, for each camera, the frame ``r - margin``
    of its released clip, read from the whole video so that the frames around
    the clip are there for the synchronisation. The person is picked on the
    clip's first frame and followed forwards, then backwards through the
    frames before the clip, where bystanders are alone in some views.
    """
    import sweep
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.paper.nlf_views import ViewsRecorder
    from rtcosmik.provenance import estimator_identity

    base = _settings()
    landmark_map = landmark_maps.select(base, subject)
    landmark_maps.require_out_of_fold(landmark_map, subject, base)
    settings = landmark_maps.apply(landmark_map, base)
    subject_dir = dataset / subject
    cameras = list(oc.CAMERAS)
    size = (settings.width, settings.height)
    static = oc.Trial(oc.STATIC, oc.session_of(subject_dir, oc.STATIC), None)
    for trial in [static] + oc.trials(subject_dir):
        views_path = out / "views" / subject / f"{trial.name}.npz"
        if views_path.exists():
            continue
        before, after = (0, 0) if trial.name == oc.STATIC else settings.opencap["sync_margin_frames"]
        cam_dir = _calibration(dataset, out, subject, trial.session)
        mtxs = load_camera_parameters(cam_dir, cameras)[0]
        estimator = sweep._nlf_estimator(mtxs, len(cameras), settings)
        estimator.reset_tracking()
        originals = [oc.original_video(subject_dir, trial.session, c, trial.name) for c in cameras]
        clips = [oc.clip_start(original, oc.video_path(subject_dir, trial.session, c, trial.name))
                 for c, original in zip(cameras, originals)]
        if len({count for _, count in clips}) != 1:
            raise RuntimeError(f"{subject}/{trial.name}: clips of different lengths {clips}")
        rows = before + clips[0][1] + after
        readers = [oc.read_frames(original, first - before, rows, size)
                   for original, (first, _) in zip(originals, clips)]
        recorder = ViewsRecorder(cameras, len(settings.marker_names), landmark_map,
                                 estimator_identity(settings, len(cameras)))
        blank = np.zeros((settings.height, settings.width, 3), np.uint8)
        absent, views = [], [None] * rows

        def estimate(row, frames):
            absent.extend((row, c) for c, frame in enumerate(frames) if frame is None)
            nlf_out = estimator.estimate_from_frames([blank if f is None else f for f in frames])[0]
            views[row] = extract_views(nlf_out, len(cameras))

        earlier = [[next(reader) for reader in readers] for _ in range(before)]
        for row in range(before, rows):
            estimate(row, [next(reader) for reader in readers])
            if row == before:
                at_clip_start = estimator.tracking_state()
        for reader in readers:
            reader.close()
        estimator.restore_tracking(at_clip_start)
        for row in reversed(range(before)):
            estimate(row, earlier[row])
        for row_views in views:
            recorder.append(row_views)
        for row, camera in absent:
            for array in (recorder.keypoints, recorder.poses3d, recorder.uncertainties):
                array[row][camera] = np.nan
        recorder.save(views_path)
        views_path.with_suffix(".json").write_text(json.dumps(
            {"clip_first_frame": [first for first, _ in clips], "clip_frames": clips[0][1],
             "margin_frames": [before, after]}))
        LOGGER.info(f"recorded {subject}/{trial.name}: {rows} rows, clip of {clips[0][1]} frames")


def _world_landmarks(poses, cam_dir, cameras):
    """Per-camera landmarks (F, C, M, 3) taken from each camera's frame to the world."""
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    projections = load_camera_parameters(cam_dir, cameras)[2]
    world_R, world_T = (np.asarray(a) for a in load_world_transformation(cam_dir, cameras[0]))
    world = []
    for camera, projection in enumerate(projections):
        projection = np.asarray(projection)
        in_reference = (poses[:, camera] - projection[:, 3]) @ projection[:, :3]
        world.append(in_reference @ world_R.T + world_T)
    return np.stack(world, 1)


def _without_far_views(views_path, cam_dir, cameras, settings):
    """``replay_markers`` edit that blanks the views holding someone else, and their share per camera.

    The five recorded cameras decide, whichever of them the arm replays.
    """
    data = np.load(views_path)
    recorded = [int(c) for c in data["cameras"]]
    far = oc.far_views(_world_landmarks(data["poses3d"].astype(float), cam_dir, recorded),
                       settings.opencap["far_view_m"])[:, [recorded.index(c) for c in cameras]]

    def edit(keypoints, poses, sigma):
        keypoints, poses, sigma = keypoints.copy(), poses.copy(), sigma.copy()
        keypoints[far], poses[far], sigma[far] = np.nan, np.nan, np.nan
        return keypoints, poses, sigma
    return edit, far.mean(0)


def sync_subject(job):
    """Synchronise the cameras of each dynamic trial and set its clock on the mocap's.

    Writes ``synced/<subject>/<trial>.npz``: the recording with one instant per
    row, cut to the mocap trial, with the mocap time of each row.
    """
    dataset, out, subject = job
    settings = _settings()
    choice = settings.opencap
    names = list(settings.marker_names)
    common = [n for n in names if n in oc.MARKERS and n not in oc.STATIC_ONLY]
    subject_dir = dataset / subject
    done = 0
    for trial in oc.trials(subject_dir):
        raw = out / "views" / subject / f"{trial.name}.npz"
        synced = _views(out, subject, trial.name)
        if synced.exists():
            continue
        data = dict(np.load(raw))
        clip = json.loads(raw.with_suffix(".json").read_text())
        margin = clip["margin_frames"][0]
        cameras = [int(c) for c in data["cameras"]]
        world = _world_landmarks(data["poses3d"].astype(float),
                                 _calibration(dataset, out, subject, trial.session), cameras)
        # Offsets from the clip itself, without the views that hold a bystander;
        # the median keeps one out of the clock too.
        # Centred on the body, so a view's depth error along a walk is not read as a lag.
        on_subject = np.where(oc.far_views(world, choice["far_view_m"])[:, :, None, None], np.nan, world)
        on_subject = on_subject - on_subject.mean(axis=2, keepdims=True)
        offsets = oc.camera_offsets(on_subject[margin:margin + clip["clip_frames"]],
                                    choice["max_camera_lag_frames"])
        world, first = oc.align_cameras(world, offsets)
        with np.errstate(all="ignore"):
            fused = np.nanmedian(world, axis=1)[:, [names.index(n) for n in common]]
        mocap_times = oc.read_trc(oc.trc_path(subject_dir, trial.name))[0]
        mocap = oc.reference_markers(oc.trc_path(subject_dir, trial.name))
        mocap = np.stack([mocap[n] for n in common], 1)
        nominal = (first - margin) / settings.fs
        offset, residual = oc.clock_offset(
            fused, settings.fs, mocap_times, mocap,
            search=(nominal + choice["clock_search_s"][0], nominal + choice["clock_search_s"][1]))
        times = np.arange(len(fused)) / settings.fs + offset
        keep = (times >= mocap_times[0]) & (times <= mocap_times[-1])
        for key in ("keypoints", "poses3d", "uncertainties"):
            data[key] = oc.align_cameras(data[key], offsets)[0][keep]
        synced.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(synced, times=times[keep], **data)
        synced.with_suffix(".json").write_text(json.dumps(
            {"camera_offsets_frames": offsets.tolist(), "clock_offset_s": offset,
             "clock_offset_from_nominal_s": offset - nominal, "residual_mm": residual * 1000,
             "far_views_share": oc.far_views(world[keep], choice["far_view_m"]).mean(0).tolist(),
             "frames": int(keep.sum()), "mocap_s": float(mocap_times[-1] - mocap_times[0])}))
        done += 1
    return subject, done


def replay_arm(job):
    """One arm on one subject: calibrated on its own static landmarks, one run per dynamic trial.

    With ``runs`` and ``calibrations`` in the job, only the trials that have a
    calibration folder ``<calibrations>/<subject>/<trial>/<tag>`` are replayed,
    under it, into ``<runs>/<subject>/<trial>/<tag>``.
    """
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.landmark_maps import checked_recording
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.provenance import recorded_estimator, start_run
    dataset, out, subject, tag = job[:4]
    runs, calibrations = job[4:] if len(job) > 4 else (out / "runs", None)
    settings = _settings()
    cameras = list(oc.ARMS[tag])
    subject_dir = dataset / subject
    todo = [t for t in oc.trials(subject_dir) if not _done(runs / subject / t.name / tag)
            and (calibrations is None or (calibrations / subject / t.name / tag).is_dir())]
    if not todo:
        return subject, tag, 0
    static_dir = _calibration(dataset, out, subject, oc.session_of(subject_dir, oc.STATIC))
    edit, _ = _without_far_views(_views(out, subject, oc.STATIC), static_dir, cameras, settings)
    static = replay_markers(_views(out, subject, oc.STATIC), static_dir, cameras,
                            fusion=settings.fusion, edit=edit)
    posture = np.median([p3d for _, p3d in static], axis=0)
    solver = HumanSolver(settings, **_anthropometry(subject_dir), logger=logging.getLogger("solve"))
    q = solver.calibrate(dict(zip(settings.marker_names, posture)))
    oc.require_standing(settings.joint_angles_names, q)
    for trial in todo:
        run_dir = runs / subject / trial.name / tag
        views = _views(out, subject, trial.name)
        landmark_map = checked_recording(views, subject, settings)
        cam_dir = (_calibration(dataset, out, subject, trial.session) if calibrations is None
                   else calibrations / subject / trial.name / tag)
        edit, far = _without_far_views(views, cam_dir, cameras, settings)
        start_run(run_dir, settings, dict(name="nlf", tag=tag, cameras=cameras, participant=subject,
                                          task=trial.name, dataset=str(dataset), views=str(views),
                                          reconstruction="fuse3d", calibration=oc.STATIC,
                                          cameras_calibration=str(cam_dir),
                                          far_views_share=far.tolist()),
                  repo=REPO, landmark_map=landmark_map, estimator=recorded_estimator(views))
        marker_filter = MarkerFilter(len(settings.marker_names), settings)
        frames = [(frame, dict(zip(settings.marker_names, marker_filter(p3d))))
                  for frame, p3d in replay_markers(views, cam_dir, cameras,
                                                   fusion=settings.fusion, edit=edit)]
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
    ap.add_argument("stage", choices=["record", "sync", "reference", "replay", "report"])
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
    elif args.stage == "sync":
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            for result in pool.imap_unordered(
                    sync_subject, [(args.dataset, args.out, s) for s in args.subjects]):
                LOGGER.info(f"  synced {result}")
    elif args.stage == "replay":
        _pool(replay_arm, [(args.dataset, args.out, s, tag) for tag in oc.ARMS
                           for s in args.subjects], args.workers)
    else:
        import opencap_report
        opencap_report.write(args.dataset, args.out, args.subjects, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
