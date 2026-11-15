#!/usr/bin/env python3
"""Human-pose calibration on COMFI against the dataset calibration.

Stages, each resumable, all under ``--out``:

    extract    NLF over each participant's walking recordings -> views/
    calibrate  extrinsics from the circular walk -> cam_params/, geometry.csv, recordings.csv
    replay     the six evaluation tasks under that calibration -> runs/, results/
    perturb    the same tasks under perturbed dataset extrinsics -> results/
    report     calibration.md (--baseline, --baseline-hand)

    python3 scripts/python/paper/calibration_eval.py extract --out /root/workspace/oof/t15
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
CAMERAS = [0, 2, 4, 6]
#: The calibration recording, and a second walk to calibrate from.
WALKS = ("CircularWalking", "StraightWalking")


def extract(out):
    """Per-camera NLF output of every walking recording, with the participant's fold map."""
    import sweep
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters
    from rtcosmik.config_loader import settings as base
    from rtcosmik.folds import PARTICIPANTS
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.paper.nlf_views import ViewsRecorder
    from rtcosmik.provenance import estimator_identity, require_clean
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    require_clean()
    for participant in PARTICIPANTS:
        landmark_map = landmark_maps.select(base, participant)
        landmark_maps.require_out_of_fold(landmark_map, participant, base)
        settings = landmark_maps.apply(landmark_map, base)
        mtxs = load_camera_parameters(DATASET / "cam_params" / participant, CAMERAS)[0]
        estimator = sweep._nlf_estimator(mtxs, len(CAMERAS), settings)
        for task in WALKS:
            path = out / "views" / participant / f"{task}.npz"
            if path.exists():
                continue
            videos = [DATASET / "videos" / participant / task / f"camera_{c}.mp4" for c in CAMERAS]
            source = OfflineVideoSource(paths=[str(v) for v in videos],
                                        size_wh=(settings.width, settings.height), loop=False)
            recorder = ViewsRecorder(CAMERAS, len(settings.marker_names), landmark_map,
                                     estimator_identity(settings, len(CAMERAS)))
            try:
                while (frames := source.read()) is not None:
                    output = estimator.estimate_from_frames(frames)[0]
                    recorder.append(extract_views(output, len(CAMERAS)))
            finally:
                source.release()
            recorder.save(path)
            print(f"{participant}/{task}: {len(recorder.keypoints)} frames", flush=True)


def reference_poses(participant):
    """Dataset intrinsics, and each camera's pose relative to camera 0."""
    from rtcosmik.camera.cam_utils import load_camera_parameters
    mtxs, dists, _, rotations, translations = load_camera_parameters(
        DATASET / "cam_params" / participant, CAMERAS)
    return mtxs, dists, [(R, T.reshape(3)) for R, T in zip(rotations, translations)]


def write_calibration(root, participant, poses=None, world=None):
    """A calibration folder: dataset intrinsics and camera 0's world pose, with either
    the stereo chain of ``poses`` or a world pose per camera (``world``)."""
    import shutil
    from rtcosmik.calibration.files import save_stereo_pose, save_world_pose
    from rtcosmik.camera.cam_utils import (cam_to_world_path, intrinsics_path, load_cam_params,
                                           load_cam_to_world)
    source = DATASET / "cam_params" / participant
    shutil.rmtree(root, ignore_errors=True)
    for camera in CAMERAS:
        target = Path(intrinsics_path(root, camera))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(intrinsics_path(source, camera), target)
    if world is None:
        world = {0: load_cam_to_world(cam_to_world_path(source, 0))}
        intrinsics = [load_cam_params(intrinsics_path(source, c)) for c in CAMERAS]
        for a in range(len(CAMERAS) - 1):
            R = poses[a + 1][0] @ poses[a][0].T
            T = poses[a + 1][1] - R @ poses[a][1]
            save_stereo_pose(*intrinsics[a], *intrinsics[a + 1], R, T, 0.0,
                             CAMERAS[a], CAMERAS[a + 1], str(root))
    for camera, (R, T) in world.items():
        save_world_pose(np.asarray(R), np.asarray(T).reshape(3), camera, str(root))


def calibrate(out, views):
    """Extrinsics from each participant's walks and their errors; which recordings are refused."""
    import csv
    from rtcosmik.calibration import human
    from rtcosmik.folds import PARTICIPANTS
    from rtcosmik.paper.calibration_study import pose_errors

    geometry, recordings = [], []
    for participant in PARTICIPANTS:
        reference = reference_poses(participant)[2]
        paths = [out / "views" / participant / f"{task}.npz" for task in WALKS]
        for path in paths + sorted((views / participant).glob("*.npz")):
            poses3d = np.load(path)["poses3d"]
            row = dict(participant=participant, task=path.stem, frames=len(poses3d),
                       travel_m=round(human.travel(poses3d[:, 0]), 3), refused="")
            recordings.append(row)
            try:
                result = human.calibrate(poses3d)
            except human.DegenerateRecording:
                row["refused"] = "yes"
                continue
            if path.stem == WALKS[0]:
                write_calibration(out / "cam_params" / participant, participant, result.poses)
            for index in range(1, len(CAMERAS)):
                errors = pose_errors(result.poses[index], reference[index])
                geometry.append(dict(participant=participant, task=path.stem,
                                     camera=CAMERAS[index],
                                     residual_mm=round(1e3 * result.residual_m[index], 1),
                                     **{k: round(v, 3) for k, v in errors.items()}))
    for name, rows in (("geometry", geometry), ("recordings", recordings)):
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


#: Perturbations of the dataset extrinsics: tag -> (rotation in degrees, shift in metres).
PERTURBATIONS = {"rot_0.5": (0.5, 0.0), "rot_1": (1.0, 0.0), "rot_2": (2.0, 0.0),
                 "rot_5": (5.0, 0.0), "shift_10": (0.0, 0.010), "shift_20": (0.0, 0.020),
                 "shift_50": (0.0, 0.050), "shift_100": (0.0, 0.100)}
HUMAN = "calib_human"


def perturbed_calibration(out, participant, task, tag):
    """The dataset calibration with cameras 2, 4 and 6 perturbed; seeded per trial."""
    import zlib
    from rtcosmik.camera.cam_utils import cam_to_world_path, load_cam_to_world
    from rtcosmik.paper.calibration_study import perturbed
    angle, shift = PERTURBATIONS[tag]
    rng = np.random.default_rng(zlib.crc32(f"{participant}/{task}/{tag}".encode()))
    source = DATASET / "cam_params" / participant
    world = {}
    for camera in CAMERAS:
        R, T = load_cam_to_world(cam_to_world_path(source, camera))
        world[camera] = (R, T) if camera == CAMERAS[0] else perturbed(R, T, angle, shift, rng)
    root = out / "perturbed" / tag / participant / task
    write_calibration(root, participant, world=world)
    return root


def replay_job(job):
    """One evaluation trial under one calibration; returns (tag, sweep row)."""
    import replay_nlf_arms
    import sweep
    from rtcosmik.config_loader import settings
    out, views, participant, task, tag = job
    arm = {"name": tag, "tag": tag, "reconstruction": "fuse3d", "cameras": CAMERAS}
    try:
        cam_dir = (out / "cam_params" / participant if tag == HUMAN
                   else perturbed_calibration(out, participant, task, tag))
        row = replay_nlf_arms.replay_trial(
            DATASET, views / participant / f"{task}.npz", out / "runs" / participant / task / tag,
            participant, task, arm, settings, cam_dir=cam_dir)
    except Exception as exc:
        row = dict.fromkeys(sweep.FIELDS, "")
        row.update(arm=tag, participant=participant, task=task,
                   status=f"{type(exc).__name__}: {exc}"[:200])
    return tag, row


def replay(out, views, tags, workers):
    """The evaluation trials under ``tags``, one summary per tag in results/."""
    import csv
    import multiprocessing
    import sweep
    from rtcosmik.provenance import require_clean
    require_clean()
    (out / "results").mkdir(parents=True, exist_ok=True)
    trials = sorted((p.parent.name, p.stem) for p in views.glob("*/*.npz"))
    jobs, handles = [], {}
    for tag in tags:
        path = out / "results" / f"{tag}.csv"
        done = ({(r["participant"], r["task"]) for r in csv.DictReader(open(path))}
                if path.exists() else set())
        fresh = not path.exists()
        handles[tag] = open(path, "a", newline="")
        if fresh:
            csv.DictWriter(handles[tag], fieldnames=sweep.FIELDS).writeheader()
        jobs += [(out, views, p, t, tag) for p, t in trials if (p, t) not in done]
    print(f"{len(jobs)} replays", flush=True)
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for tag, row in pool.imap_unordered(replay_job, jobs):
            csv.DictWriter(handles[tag], fieldnames=sweep.FIELDS).writerow(row)
            handles[tag].flush()
            if row["status"] != "ok":
                print(f"  {tag} {row['participant']}/{row['task']}: {row['status']}", flush=True)
    for handle in handles.values():
        handle.close()


def _by_participant(frame, columns):
    return frame.groupby("participant")[columns].mean()


def report(out, baseline, baseline_hand):
    """calibration.md: geometry, refusals, and errors under each calibration."""
    import pandas as pd
    geometry = pd.read_csv(out / "geometry.csv")
    recordings = pd.read_csv(out / "recordings.csv")
    geometry["pair"] = np.where(geometry.camera == 2, "adjacent", "opposite")
    lines = ["# Human-pose calibration on COMFI", "",
             "## Extrinsics against the dataset calibration (median, quartiles, maximum)", "",
             "| Recording | Pair | Cameras | Baseline (m) | Rotation (deg) | Position (mm) | "
             "Length (%) |", "|---|---|---|---|---|---|---|"]

    def spread(values):
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        return f"{q2:.2f} ({q1:.2f}, {q3:.2f}; {np.max(values):.2f})"

    for (task, pair), g in geometry[geometry.task.isin(WALKS)].groupby(["task", "pair"]):
        lines.append(f"| {task} | {pair} | {len(g)} | {g.baseline_m.median():.2f} | "
                     f"{spread(g.rotation_deg)} | {spread(g.position_mm)} | "
                     f"{spread(g.length_pct.abs())} |")
    lines += ["", "## Recordings refused", "", "| Recording | Trials | Refused | Travel (m) |",
              "|---|---|---|---|"]
    for task, g in recordings.groupby("task"):
        lines.append(f"| {task} | {len(g)} | {(g.refused == 'yes').sum()} | "
                     f"{g.travel_m.min():.2f} to {g.travel_m.max():.2f} |")

    columns = ["causal_rmse_mean", "causal_marker_mm"]
    keys = ["participant", "task"]
    reference = pd.read_csv(baseline)
    lines += ["", "## Evaluation tasks under each calibration (mean over participants)", "",
              "| Calibration | Trials | Joint angles (deg) | Difference (deg, mean; largest "
              "participant) | Markers (mm) | Worst participant (mm) |", "|---|---|---|---|---|---|"]
    rows = [("dataset", reference)] + [(tag, pd.read_csv(out / "results" / f"{tag}.csv"))
                                       for tag in [HUMAN] + list(PERTURBATIONS)]
    for tag, frame in rows:
        both = frame[frame.status == "ok"].merge(reference, on=keys, suffixes=("", "_ref"))
        mean = _by_participant(both, columns + [c + "_ref" for c in columns])
        difference = mean.causal_rmse_mean - mean.causal_rmse_mean_ref
        lines.append(f"| {tag} | {len(both)} | {mean.causal_rmse_mean.mean():.2f} | "
                     f"{difference.mean():+.3f}; {difference.abs().max():.2f} | "
                     f"{mean.causal_marker_mm.mean():.1f} | {mean.causal_marker_mm.max():.0f} |")

    hands = [pd.read_csv(f) for f in (baseline_hand, out / "hand" / "pooled.csv") if f.exists()]
    if hands:
        hand = pd.concat(hands)
        hand = hand[hand.task == "all"]
        lines += ["", "## Hand-to-robot distance on the robot tasks", "",
                  "| Calibration | Hand | Trials | Bias (mm) | SD (mm) | 95th pct. of abs. error "
                  "(mm) | Missed at 300 mm (%) | False alarms at 300 mm (%) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in hand.iterrows():
            lines.append(f"| {r.arm} | {r.hand} | {r.trials} | {r.causal_bias_mm:+.1f} | "
                         f"{r.causal_sd_mm:.1f} | {r.causal_abs_p95_mm:.1f} | "
                         f"{r.causal_300mm_missed_pct:.2f} | "
                         f"{r.causal_300mm_false_alarm_pct:.2f} |")
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["extract", "calibrate", "replay", "perturb", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", type=Path, help="recorded views of the evaluation trials")
    parser.add_argument("--baseline", type=Path, help="summary of the arm under the dataset calibration")
    parser.add_argument("--baseline-hand", type=Path, help="its pooled hand-distance table")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.out)
    elif args.stage == "calibrate":
        calibrate(args.out, args.views)
    elif args.stage == "report":
        report(args.out, args.baseline, args.baseline_hand)
    elif args.stage == "replay":
        replay(args.out, args.views, [HUMAN], args.workers)
    else:
        replay(args.out, args.views, list(PERTURBATIONS), args.workers)
    print("done", flush=True)


if __name__ == "__main__":
    main()
