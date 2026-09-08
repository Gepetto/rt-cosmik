#!/usr/bin/env python3
"""Which two of the four cameras make the best pair?

The 2-camera configuration needs a pair chosen on evidence rather than by
convention, so this scores all six from {0, 2, 4, 6} on the same trials the
other studies use. The mmpose arm is used because its 2D is precomputed and so
the sweep is cheap; the geometry it measures is a property of the rig, not of
the pose estimator, so the answer carries over.

    python3 scripts/python/paper/study_camera_pairs.py --dataset /root/workspace/COMFI

Reports the mean per pair and also the per-task spread, because the best pair on
average need not be the best for a subject walking away from half the rig.
"""
import argparse
import itertools
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

import sweep as sweep_mod
from study_horizon import DEFAULT_TRIALS

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s", force=True)
LOGGER = logging.getLogger("pairs")
LOGGER.setLevel(logging.INFO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--summary", default="results/camera_pairs.csv")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = sweep_mod._load_eval()

    pairs = list(itertools.combinations(args.cameras, 2))
    results = {}
    for pair in pairs:
        label = "-".join(map(str, pair))
        scores = {}
        for participant, task in DEFAULT_TRIALS:
            out_dir = Path(settings.output_dir) / participant / task / f"pair_{label}"
            try:
                sweep_mod.run_mmpose(args.dataset, participant, task, list(pair),
                                     out_dir, settings)
                row = sweep_mod.score(
                    out_dir,
                    Path(args.dataset) / "mocap" / "aligned" / participant / task, ev)
                scores[(participant, task)] = row["joint_rmse_mean"]
            except Exception as exc:
                LOGGER.warning(f"  {label} {participant}/{task} failed: {exc}")
        results[label] = scores
        if scores:
            LOGGER.info(f"pair {label}: mean {np.mean(list(scores.values())):.3f} deg "
                        f"over {len(scores)} trials")

    shared = set.intersection(*(set(v) for v in results.values() if v))
    labels = [l for l in results if results[l]]
    print(f"\n{'trial':<28}" + "".join(f"{l:>10}" for l in labels))
    print("-" * (28 + 10 * len(labels)))
    for key in sorted(shared):
        line = f"{'/'.join(key):<28}"
        best = min(results[l][key] for l in labels)
        for l in labels:
            v = results[l][key]
            line += f"{v:>9.1f}" + ("*" if v == best else " ")
        print(line)
    print("-" * (28 + 10 * len(labels)))
    means = {l: np.mean([results[l][k] for k in shared]) for l in labels}
    print(f"{'mean':<28}" + "".join(f"{means[l]:>10.2f}" for l in labels) + "   deg")
    order = sorted(means, key=means.get)
    print(f"\nbest pair: {order[0]} ({means[order[0]]:.2f} deg); "
          f"worst: {order[-1]} ({means[order[-1]]:.2f} deg)")
    print("* marks the best pair for that trial")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as handle:
        handle.write("pair," + ",".join("/".join(k) for k in sorted(shared)) + ",mean\n")
        for l in labels:
            handle.write(l + "," + ",".join(f"{results[l][k]:.3f}" for k in sorted(shared))
                         + f",{means[l]:.3f}\n")
    print(f"written to {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
