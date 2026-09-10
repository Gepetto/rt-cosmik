#!/usr/bin/env python3
"""Paired differences between arms, over the trials they all completed.

Every arm ran the same trials, so the arms are paired and the unpaired spread
across trials is the wrong yardstick: a task that is hard for one arm is hard for
all of them, and that shared difficulty cancels in a difference. Reporting the
per-trial spread would make effects of a degree or two look like noise; reporting
only the standard error would hide how often individual trials disagree with the
average. Both are printed.

    python3 scripts/python/paper/study_paired.py                 # full matrix
    python3 scripts/python/paper/study_paired.py --against fastsam_0
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

ARMS = [
    ("mmpose_0-2", "mmpose 2 cams"),
    ("mmpose_0-2-4-6", "mmpose 4 cams"),
    ("nlf2d_0-2", "NLF-2D tri 2 cams"),
    ("nlf2d_0-2-4-6", "NLF-2D tri 4 cams"),
    ("nlf_0", "NLF-3D 1 cam"),
    ("nlf_0-2", "NLF-3D 2 cams"),
    ("nlf_0-2-4-6", "NLF-3D 4 cams"),
    ("fastsam_0", "FastSAM-3D 1 cam"),
]


def load(metric):
    """{tag: {(participant, task): value}} for every arm with a summary."""
    out, labels = {}, {}
    for tag, label in ARMS:
        path = REPO / "results" / "vs_mocap" / f"{tag}.csv"
        if not path.exists():
            continue
        values = {}
        for row in csv.DictReader(open(path)):
            if row.get("status") != "ok" or not row.get(metric):
                continue
            values[(row["participant"], row["task"])] = float(row[metric])
        if values:
            out[tag] = values
            labels[tag] = label
    return out, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="joint_rmse_mean",
                    help="summary column to compare (default joint_rmse_mean)")
    ap.add_argument("--against", default=None,
                    help="compare every arm against this one instead of all pairs")
    args = ap.parse_args()

    data, labels = load(args.metric)
    if len(data) < 2:
        raise SystemExit("need at least two rescored summaries under results/vs_mocap/")
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if not common:
        raise SystemExit("the arms share no completed trials")

    tags = [t for t, _ in ARMS if t in data]
    series = {t: np.array([data[t][k] for k in common]) for t in tags}

    print(f"Metric: {args.metric}   Trials common to all arms: {len(common)}\n")
    width = max(len(labels[t]) for t in tags) + 2
    print(f"{'arm':<{width}}{'mean':>9}{'std':>8}")
    for t in tags:
        print(f"{labels[t]:<{width}}{series[t].mean():>9.2f}{series[t].std():>8.2f}")

    pairs = ([(b, args.against) for b in tags if b != args.against]
             if args.against else
             [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:]])
    if args.against and args.against not in series:
        raise SystemExit(f"no summary for {args.against}")

    print(f"\nPaired differences over the same {len(common)} trials.")
    print("Negative means the first arm is better. SE is the standard error of")
    print("the paired difference; 'wins' is how often it is actually lower.\n")
    print(f"{'comparison':<44}{'diff':>9}{'SE':>8}{'spread':>9}{'wins':>10}")
    for a, b in pairs:
        d = series[a] - series[b]
        se = d.std(ddof=1) / np.sqrt(len(d))
        wins = (d < 0).mean() * 100
        print(f"{labels[a] + ' - ' + labels[b]:<44}{d.mean():>+9.2f}{se:>8.2f}"
              f"{d.std():>9.2f}{wins:>9.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
