#!/usr/bin/env python3
"""Hand-to-end-effector distance statistics of arms on the robot tasks (ticket 13).

Per trial and hand: signed error of the distance from the hand centre to the Panda's
hand frame, its upper percentiles, and missed close approaches at
``hand_distance.THRESHOLDS_MM``, causal and aligned. Pooled over trials per arm, with
the keep-or-drop rule of docs/tickets/13-robot-distance.md applied to the causal figures.

    python3 scripts/python/paper/hand_distance.py --runs /root/workspace/oof/runs \\
        --out /root/workspace/oof/t13 --arms nlf_0-2-4-6 fusion_mean
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

import robot_distance as rd
from rtcosmik.paper import hand_distance as hd
from rtcosmik.paper import scoring

HANDS = ("left", "right")
WORKERS = 8
ID_FIELDS = ["arm", "participant", "task", "hand"]


def hand_centres(skeleton, run_dir):
    """(F, 2, 3) left and right hand centres by video frame, NaN where the run has none."""
    import pandas as pd
    frames = pd.read_csv(run_dir / "markers.csv", usecols=["Frame_0"])["Frame_0"].to_numpy(int)
    index = [rd.SEGMENT_NAMES.index(f"{h}_hand") for h in HANDS]
    out = np.full((frames.max() + 1, 2, 3), np.nan)
    for row, frame in enumerate(frames[:len(skeleton.q)]):
        if np.isfinite(skeleton.q[row]).all():
            out[frame] = skeleton.segments(row)[index].mean(axis=1)
    return out


def effector_positions(robot, by_frame):
    """(F, 3) origin of the Panda's hand frame by video frame, NaN without a robot state."""
    out = np.full((max(by_frame) + 1, 3), np.nan)
    for frame, joints in by_frame.items():
        out[frame] = robot.place(joints)[1]
    return out


def one_trial(job):
    """Every arm of one trial: (rows, {(arm, hand): per-frame distances}, messages)."""
    import yaml
    from rtcosmik.config_loader import settings
    runs_root, reference_root, participant, task, arms, frames_dir = job
    by_frame = rd.robot_frames(participant, task)
    if not by_frame:
        return [], {}, [f"  {participant}/{task}: no robot states"]
    meta = yaml.safe_load((rd.DATASET / "metadata" / f"{participant}.yaml").read_text())
    effector = effector_positions(rd.Robot(participant), by_frame)
    ref_dir = reference_root / participant / task / scoring.REFERENCE_TAG
    reference = hand_centres(rd.Skeleton(ref_dir, meta), ref_dir)
    offset = scoring.clock_offset(settings, participant, task)

    rows, series, messages = [], {}, []
    for arm in arms:
        run_dir = runs_root / participant / task / arm
        try:
            lag = scoring.joint_lag(run_dir, ref_dir)
            estimate = hand_centres(rd.Skeleton(run_dir, meta), run_dir)
        except Exception as exc:
            messages.append(f"  {arm} {participant}/{task}: {type(exc).__name__}: {exc}")
            continue
        arrays = {}
        for k, hand in enumerate(HANDS):
            row, distances = hd.score_hand(estimate[:, k], reference[:, k], effector, offset, lag)
            rows.append({"arm": arm, "participant": participant, "task": task, "hand": hand, **row})
            series[arm, hand] = distances
            arrays.update({f"{hand}_{key}_mm": d for key, d in distances.items()})
        target = frames_dir / arm
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"{participant}_{task}.npz", clock_offset=offset, lag=lag,
                            **arrays)
    return rows, series, messages


def write(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True,
                    help="root of the arms' run folders: <participant>/<task>/<arm>")
    ap.add_argument("--reference", type=Path, default=rd.DATASET / "output" / "campaign",
                    help="root of the reference runs: <participant>/<task>/mocap_reference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="run folder names")
    args = ap.parse_args()

    jobs = []
    for participant in sorted(p.name for p in args.runs.iterdir() if p.is_dir()):
        for task in rd.ROBOT_TASKS:
            arms = [a for a in args.arms
                    if (args.runs / participant / task / a / "joint_angles.csv").exists()]
            if arms:
                jobs.append((args.runs, args.reference, participant, task, arms,
                             args.out / "frames"))

    rows, series = [], {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for trial_rows, trial_series, messages in pool.map(one_trial, jobs):
            for m in messages:
                print(m, flush=True)
            rows += trial_rows
            for (arm, hand), distances in trial_series.items():
                task = trial_rows[0]["task"]
                for key in ((arm, hand, "all"), (arm, hand, task)):
                    series.setdefault(key, []).append(distances)

    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        write(args.out / f"{arm}.csv", ID_FIELDS + hd.ROW_FIELDS,
              [r for r in rows if r["arm"] == arm])
    pooled = []
    for (arm, hand, task), trials in sorted(series.items()):
        row = hd.pool(trials)
        pooled.append({"arm": arm, "hand": hand, "task": task, "trials": len(trials), **row,
                       **hd.decide(row)})
    fields = ["arm", "hand", "task", "trials"] + hd.POOLED_FIELDS + list(hd.decide(pooled[0]))
    write(args.out / "pooled.csv", fields, pooled)
    for arm in args.arms:
        hands = [r for r in pooled if r["arm"] == arm and r["task"] == "all"]
        verdict = "keep" if hands and all(r["keep"] for r in hands) else "drop"
        print(f"{arm}: {hands[0]['trials'] if hands else 0} trials, rule says {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
