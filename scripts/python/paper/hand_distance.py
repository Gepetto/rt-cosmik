#!/usr/bin/env python3
"""Hand-to-end-effector distance statistics of arms on the robot tasks.

Per trial and hand: signed error of the distance from the hand centre to the Panda's
hand frame, its upper percentiles, and missed close approaches at
``hand_distance.THRESHOLDS_MM``, causal and aligned. Pooled over trials per arm, with
the keep-or-drop rule of docs/tickets/13-robot-distance.md applied to the causal figures.
Under ``<out>/``: ``<arm>.csv`` per trial and hand, ``pooled.csv``, ``skipped.csv`` (trials
and hands left out, with the reason) and ``frames/`` (per-frame distances).

    python3 scripts/python/paper/hand_distance.py --runs /root/workspace/oof/runs \\
        --dataset /root/workspace/COMFI --out /root/workspace/oof/t13 \\
        --arms nlf_0-2-4-6 fusion_mean
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
SKIP_FIELDS = ["participant", "task", "arm", "hand", "reason"]
VERDICT_FIELDS = ["keep", "missed_threshold_mm", "missed_pct", "abs_p99_ok", "over_p95_ok",
                  "missed_ok"]


def skip(participant, task, reason, arm="all", hand="all"):
    return dict(zip(SKIP_FIELDS, (participant, task, arm, hand, reason)))


def hand_centres(run_dir, meta):
    """(F, 2, 3) left and right hand centres by video frame, NaN where the run has none."""
    skeleton = rd.Skeleton(run_dir, meta)
    skeleton.q = scoring.by_frame(run_dir)[1]
    index = [rd.SEGMENT_NAMES.index(f"{h}_hand") for h in HANDS]
    out = np.full((len(skeleton.q), 2, 3), np.nan)
    for frame in np.flatnonzero(np.isfinite(skeleton.q).all(1)):
        out[frame] = skeleton.segments(frame)[index].mean(axis=1)
    return out


def effector_positions(robot, by_frame):
    """(F, 3) origin of the Panda's hand frame by video frame, NaN without a robot state."""
    out = np.full((max(by_frame) + 1, 3), np.nan)
    for frame, joints in by_frame.items():
        out[frame] = robot.place(joints)[1]
    return out


def one_trial(job):
    """Every arm of one trial: (rows, {(arm, hand): per-frame distances}, skipped)."""
    import yaml
    from rtcosmik.config_loader import settings
    runs_root, dataset, participant, task, arms, frames_dir = job
    why = lambda exc: f"{type(exc).__name__}: {exc}"
    by_frame = rd.robot_frames(participant, task, dataset)
    if not by_frame:
        return [], {}, [skip(participant, task, "no robot states")]
    ref_dir = scoring.reference_dir(dataset, participant, task)
    if not (ref_dir / "joint_angles.csv").exists():
        return [], {}, [skip(participant, task, "no reference run")]
    try:
        meta = yaml.safe_load((dataset / "metadata" / f"{participant}.yaml").read_text())
        reference = hand_centres(ref_dir, meta)
        effector = effector_positions(rd.Robot(participant, dataset), by_frame)
        offset = scoring.clock_offset(settings, participant, task)
    except Exception as exc:
        return [], {}, [skip(participant, task, why(exc))]

    rows, series, skipped = [], {}, []
    for arm in arms:
        run_dir = runs_root / participant / task / arm
        try:
            lag = scoring.joint_lag(run_dir, ref_dir)
            estimate = hand_centres(run_dir, meta)
        except Exception as exc:
            skipped.append(skip(participant, task, why(exc), arm))
            continue
        arrays = {}
        for k, hand in enumerate(HANDS):
            try:
                row, distances = hd.score_hand(estimate[:, k], reference[:, k], effector, offset,
                                               lag)
            except ValueError as exc:
                skipped.append(skip(participant, task, str(exc), arm, hand))
                continue
            rows.append({"arm": arm, "participant": participant, "task": task, "hand": hand, **row})
            series[arm, hand] = distances
            arrays.update({f"{hand}_{key}_mm": d for key, d in distances.items()})
        if arrays:
            target = frames_dir / arm
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / f"{participant}_{task}.npz", clock_offset=offset,
                                lag=lag, **arrays)
    return rows, series, skipped


def write(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out, arms, rows, series, skipped):
    """The CSV files; ``series`` is ``{(arm, hand, task or "all"): [per-frame distances]}``.

    Returns the pooled rows. ``skipped`` in pooled.csv counts what that row leaves out.
    """
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        write(out / f"{arm}.csv", ID_FIELDS + hd.ROW_FIELDS, [r for r in rows if r["arm"] == arm])
    write(out / "skipped.csv", SKIP_FIELDS, skipped)
    pooled = []
    for (arm, hand, task), trials in sorted(series.items()):
        row = hd.pool(trials)
        left_out = sum(s["arm"] in ("all", arm) and s["hand"] in ("all", hand)
                       and task in ("all", s["task"]) for s in skipped)
        pooled.append({"arm": arm, "hand": hand, "task": task, "trials": len(trials),
                       "skipped": left_out, **row, **hd.decide(row)})
    write(out / "pooled.csv", ["arm", "hand", "task", "trials", "skipped"] + hd.POOLED_FIELDS
          + VERDICT_FIELDS, pooled)
    return pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True,
                    help="root of the arms' run folders: <participant>/<task>/<arm>")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset root holding the robot states and the reference runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="run folder names")
    args = ap.parse_args()

    jobs = []
    for participant in sorted(p.name for p in args.runs.iterdir() if p.is_dir()):
        for task in rd.ROBOT_TASKS:
            arms = [a for a in args.arms
                    if (args.runs / participant / task / a / "joint_angles.csv").exists()]
            if arms:
                jobs.append((args.runs, args.dataset, participant, task, arms,
                             args.out / "frames"))

    rows, series, skipped = [], {}, []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for job, (trial_rows, trial_series, trial_skipped) in zip(jobs, pool.map(one_trial, jobs)):
            for s in trial_skipped:
                print("  skipped " + " ".join(s.values()), flush=True)
            rows += trial_rows
            skipped += trial_skipped
            for (arm, hand), distances in trial_series.items():
                for key in ((arm, hand, "all"), (arm, hand, job[3])):
                    series.setdefault(key, []).append(distances)

    pooled = write_outputs(args.out, args.arms, rows, series, skipped)
    for arm in args.arms:
        hands = {r["hand"]: r for r in pooled if r["arm"] == arm and r["task"] == "all"}
        if set(hands) != set(HANDS):
            print(f"{arm}: no scored trial for {sorted(set(HANDS) - set(hands))}, no verdict",
                  flush=True)
            continue
        verdict = "keep" if all(r["keep"] for r in hands.values()) else "drop"
        trials = ", ".join(f"{h} {r['trials']} trials ({r['skipped']} skipped)"
                           for h, r in hands.items())
        print(f"{arm}: {trials}, rule says {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
