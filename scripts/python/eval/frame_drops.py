#!/usr/bin/env python3
"""Where did the pipeline miss camera frames, and did it settle down?

The per-camera frame counters recorded in markers.csv say exactly which frames
were processed, so the gaps between consecutive rows are the frames that were
missed. A summary over the whole run cannot tell a warm-up cost from a recurring
stall -- that difference decides whether anything needs fixing at all -- so this
splits the run into deciles and shows where the losses actually are.

    python3 scripts/python/eval/frame_drops.py output/test/markers.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd


def analyse(path, big=2):
    df = pd.read_csv(path)
    counters = [c for c in df.columns if c.startswith("Frame_")]
    if not counters:
        raise SystemExit(f"{path} has no Frame_* counter columns")
    frames = df[counters].to_numpy()
    if len(frames) < 3:
        raise SystemExit(f"{path} has only {len(frames)} rows; is a run still writing it?")

    steps = np.diff(frames[:, 0]).astype(int)
    total = int(frames[-1, 0] - frames[0, 0])
    skew = int(np.abs(frames - frames[:, :1]).max())

    print(f"{path}: {len(df)} rows, counters {frames[0,0]} -> {frames[-1,0]}")
    print(f"  cameras advanced {total} frames; {len(df)} processed "
          f"({100*len(df)/max(total,1):.0f}%)")
    print(f"  worst inter-camera skew within a row: {skew} frame(s)")
    print(f"  steps: median {np.median(steps):.0f}, mean {steps.mean():.2f}, "
          f"max {steps.max()}, negative-or-zero {(steps<=0).sum()}")

    n = len(steps)
    print(f"\n  {'decile':<10}{'rows':>7}{'mean step':>11}{'kept %':>9}"
          f"{f'skips>{big}':>10}{'worst':>7}")
    kept = []
    for k in range(10):
        seg = steps[k * n // 10:(k + 1) * n // 10]
        if not len(seg):
            continue
        kept.append(100 / seg.mean())
        print(f"  {k*10:>3}-{(k+1)*10:<6}{len(seg):>7}{seg.mean():>11.2f}"
              f"{100/seg.mean():>9.0f}{(seg>big).sum():>10}{seg.max():>7}")

    head, tail = steps[:n // 10], steps[n // 10:]
    if len(head) and len(tail):
        print(f"\n  first 10%: {100/head.mean():.0f}% kept | "
              f"rest: {100/tail.mean():.0f}% kept")
        # A warm-up cost is worth ignoring; a flat profile means the stalls are
        # part of steady-state behaviour and worth chasing.
        if head.mean() > 1.5 * tail.mean():
            print("  -> losses are concentrated at the start (warm-up), and the "
                  "run settles down")
        elif kept and max(kept) - min(kept) < 10:
            print("  -> losses are spread evenly: this is steady-state, not warm-up")
        else:
            print("  -> losses are uneven but not a warm-up; look at the decile "
                  "with the worst kept %")
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="markers.csv from one or more runs")
    ap.add_argument("--big", type=int, default=2,
                    help="a step larger than this counts as a real skip")
    args = ap.parse_args()
    for path in args.csv:
        analyse(path, args.big)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
