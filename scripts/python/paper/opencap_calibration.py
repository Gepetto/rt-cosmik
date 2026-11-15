#!/usr/bin/env python3
"""Human-pose calibration on OpenCap against the dataset's checkerboard calibration.

Each walking trial gets extrinsics from the other walking trials of its subject,
is replayed under them, and is scored like the checkerboard arm. ``--eval`` is the
folder of opencap_eval.py (synced views, reference runs, results); ``--out`` gets
calibration/, runs/, geometry.csv, recordings.csv, trials.csv and calibration.md.

    python3 scripts/python/paper/opencap_calibration.py calibrate --dataset D --eval E --out O
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np
import pandas as pd

from rtcosmik.paper import opencap as oc

WALKING = "walking"


def _poses3d(evaluation, subject, trial):
    return np.load(evaluation / "synced" / subject / f"{trial}.npz")["poses3d"].astype(float)


def _world_poses(cam_dir):
    """Checkerboard world pose ``(R, T)`` of each camera, ``p_world = R @ p_camera + T``."""
    from rtcosmik.camera.cam_utils import cam_to_world_path, load_cam_to_world
    return {c: tuple(np.asarray(a, float).reshape(shape) for a, shape in
                     zip(load_cam_to_world(cam_to_world_path(cam_dir, c)), ((3, 3), (3,))))
            for c in oc.CAMERAS}


def _write(root, board_dir, reference, poses):
    """Calibration folder: the board's intrinsics, the board's world pose of ``reference``,
    and every other camera placed from it by ``poses`` (camera -> (R, T) relative to it)."""
    import shutil
    from rtcosmik.calibration.files import save_world_pose
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(board_dir / "intrinsics", root / "intrinsics")
    world_R, world_T = _world_poses(board_dir)[reference]
    for camera, (R, T) in poses.items():
        save_world_pose(world_R @ R.T, world_T - world_R @ R.T @ T, camera, str(root))


def calibrate(dataset, evaluation, out):
    """Extrinsics of every walking trial from the subject's other walks; errors against the board."""
    import opencap_eval
    from rtcosmik.calibration import human
    from rtcosmik.paper.calibration_study import pose_errors

    geometry, recordings = [], []
    for subject in oc.SUBJECTS:
        trials = oc.trials(dataset / subject)
        for trial in trials:
            poses3d = _poses3d(evaluation, subject, trial.name)
            row = dict(subject=subject, trial=trial.name, activity=trial.activity,
                       frames=len(poses3d), travel_m=round(human.travel(poses3d[:, 0]), 3),
                       refused="")
            recordings.append(row)
            try:
                human.calibrate(poses3d)
            except human.DegenerateRecording:
                row["refused"] = "yes"
        walks = [t for t in trials if t.activity == WALKING]
        for trial in walks:
            board_dir = opencap_eval._calibration(dataset, evaluation, subject, trial.session)
            board = _world_poses(board_dir)
            others = np.concatenate([_poses3d(evaluation, subject, t.name)
                                     for t in walks if t.name != trial.name])
            for tag, cameras in oc.ARMS.items():
                order = [cameras[0]] + [c for c in oc.CAMERAS if c != cameras[0]]
                # The released clips cover a short stretch of each walk: measured, not refused.
                result = human.calibrate(others[:, order], min_travel=0.0)
                _write(out / "calibration" / subject / trial.name / tag, board_dir, cameras[0],
                       dict(zip(order, result.poses)))
                if tag != "nlf_5cam":
                    continue
                R0, T0 = board[cameras[0]]
                for camera, pose, residual in zip(order[1:], result.poses[1:],
                                                  result.residual_m[1:]):
                    R, T = board[camera]
                    errors = pose_errors(pose, (R.T @ R0, R.T @ (T0 - T)))
                    geometry.append(dict(subject=subject, trial=trial.name, camera=camera,
                                         frames=len(others),
                                         travel_m=round(result.travel_m, 3),
                                         residual_mm=round(1e3 * residual, 1),
                                         **{k: round(v, 3) for k, v in errors.items()}))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geometry).to_csv(out / "geometry.csv", index=False)
    pd.DataFrame(recordings).to_csv(out / "recordings.csv", index=False)


def replay(dataset, evaluation, out, workers):
    import opencap_eval
    from rtcosmik.provenance import require_clean
    require_clean()
    opencap_eval._pool(opencap_eval.replay_arm,
                       [(dataset, evaluation, subject, tag, out / "runs", out / "calibration")
                        for tag in oc.ARMS for subject in oc.SUBJECTS], workers)


def score(dataset, evaluation, out):
    """Walking trials under the human-pose calibration, scored as opencap_report scores them."""
    import opencap_report
    from rtcosmik.paper import scoring
    settings = opencap_report._settings()
    max_lag = round(settings.opencap["max_lag_s"] * settings.fs)
    frames = json.loads((evaluation / "results" / "session_frames.json").read_text())
    rows = []
    for subject in oc.SUBJECTS:
        own = opencap_report.session_frames(dataset, out, subject, references=evaluation)
        for trial in oc.trials(dataset / subject):
            reference = evaluation / "runs" / subject / trial.name / scoring.REFERENCE_TAG
            for tag in oc.ARMS:
                run = out / "runs" / subject / trial.name / tag
                if not opencap_report._complete(run):
                    continue
                times = np.load(reference / "frame_times.npy")
                released = oc.read_mot(oc.released_ik(dataset / subject, trial.name))[0]
                window = oc.window_frames(times, (released[0], released[-1]))
                frame = frames[subject][str(trial.session)]
                row, _ = scoring.score_trial(
                    run, reference, settings.paper["clock_offset"], unscored=oc.UNSCORED,
                    max_lag=max_lag, frames=window, marker_frame=(frame["R"], frame["t"]))
                refitted, _ = scoring.score_trial(
                    run, reference, settings.paper["clock_offset"], unscored=oc.UNSCORED,
                    max_lag=max_lag, frames=window, marker_frame=own[trial.session])
                rows.append(dict(arm=tag, subject=subject, trial=trial.name, **row,
                                 refitted_marker_mm=refitted["causal_marker_mm"]))
    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)


def report(evaluation, out):
    """calibration.md: extrinsic differences, refusals, and errors under both calibrations."""
    geometry = pd.read_csv(out / "geometry.csv")
    recordings = pd.read_csv(out / "recordings.csv")
    human = pd.read_csv(out / "trials.csv")
    board = pd.read_csv(evaluation / "results" / "trials.csv")
    board = board[board.scope == "window"]
    both = human.merge(board, on=["arm", "subject", "trial"], suffixes=("_human", "_board"))
    lines = ["# Human-pose calibration on OpenCap", "",
             "Extrinsics of each walking trial from the subject's other walking clips, against "
             "the dataset's checkerboard calibration; then the walking trials replayed under "
             "them. The travel rule is not applied to these calibrations (the clips are short); "
             "the refusal table shows what it would refuse, trial by trial. Produced by "
             "`scripts/python/paper/opencap_calibration.py`; discussion in "
             "`resubmission-findings.md`, section 0f.", "",
             "## Extrinsics against the checkerboard (walking session, relative to Cam0)", "",
             "| Camera | Subjects | Rotation (deg) | Position (mm) | Baseline (m) | "
             "Length, absolute error (%) |",
             "|---|---|---|---|---|---|"]
    per_subject = geometry.groupby(["camera", "subject"]).mean(numeric_only=True).reset_index()
    for camera, g in per_subject.groupby("camera"):
        lines.append(f"| Cam{camera} | {len(g)} | {g.rotation_deg.median():.2f} "
                     f"({g.rotation_deg.min():.2f} to {g.rotation_deg.max():.2f}) | "
                     f"{g.position_mm.median():.0f} ({g.position_mm.min():.0f} to "
                     f"{g.position_mm.max():.0f}) | {g.baseline_m.median():.2f} | "
                     f"{g.length_pct.abs().median():.1f} |")
    lines += ["", "## Recordings refused", "",
              "| Activity | Trials | Refused | Travel (m) |", "|---|---|---|---|"]
    for activity, g in recordings.groupby("activity"):
        lines.append(f"| {activity} | {len(g)} | {(g.refused == 'yes').sum()} | "
                     f"{g.travel_m.min():.2f} to {g.travel_m.max():.2f} |")
    lines += ["", "## Walking trials under both calibrations (mean over subjects)", "",
              "| Arm | Subjects | Trials | Joint angles, board (deg) | human (deg) | "
              "Markers, board (mm) | human (mm) | human, frame refitted (mm) |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, g in both.groupby("arm"):
        s = g.groupby("subject").mean(numeric_only=True)
        lines.append(f"| {arm} | {len(s)} | {len(g)} | {s.causal_rmse_mean_board.mean():.2f} | "
                     f"{s.causal_rmse_mean_human.mean():.2f} | "
                     f"{s.causal_marker_mm_board.mean():.1f} | "
                     f"{s.causal_marker_mm_human.mean():.1f} | {s.refitted_marker_mm.mean():.1f} |")
    lines += ["", "Markers are compared in the mocap frame through one rigid transform per session, "
              "fitted on the five-camera arm: under the checkerboard calibration, or refitted "
              "under the human-pose calibration (last column)."]
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["calibrate", "replay", "report"])
    parser.add_argument("--dataset", type=Path, required=True, help="LabValidation_withVideos")
    parser.add_argument("--eval", type=Path, required=True, help="output folder of opencap_eval.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "calibrate":
        calibrate(args.dataset, args.eval, args.out)
    elif args.stage == "replay":
        replay(args.dataset, args.eval, args.out, args.workers)
    else:
        score(args.dataset, args.eval, args.out)
        report(args.eval, args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
