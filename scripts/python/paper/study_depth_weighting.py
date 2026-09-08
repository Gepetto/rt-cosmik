#!/usr/bin/env python3
"""Does range-aware view weighting help the triangulation?

The old cosmik_vizu pipeline blended two stereo pairs with a weight that mixed
detector confidence and a Gaussian proximity kernel. The proximity half of that
is the part worth keeping -- triangulated depth error grows with range, and our
weighting looks only at confidence -- but the Gaussian form has a length scale to
tune, and that pipeline's apparent gain is confounded by a Kalman filter tuned on
mocap, so the idea has never actually been measured on its own.

This measures it: the same trials, the same everything else, with per-camera
sigma either left as 1/confidence or multiplied by the distance from that camera
to the joint. Distances come from the previous frame, so nothing here is
non-causal and no ground truth is involved.

    python3 scripts/python/paper/study_depth_weighting.py --dataset /root/workspace/COMFI
"""
import argparse
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
LOGGER = logging.getLogger("depth")
LOGGER.setLevel(logging.INFO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--summary", default="results/depth_weighting.csv")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = sweep_mod._load_eval()

    per_mode = {}
    for depth_aware in (False, True):
        label = "distance" if depth_aware else "confidence"
        scores = {}
        for participant, task in DEFAULT_TRIALS:
            out_dir = (Path(settings.output_dir) / participant / task
                       / f"depth_{label}")
            try:
                sweep_mod.run_mmpose(args.dataset, participant, task, args.cameras,
                                     out_dir, settings, depth_aware=depth_aware)
                row = sweep_mod.score(
                    out_dir,
                    Path(args.dataset) / "mocap" / "aligned" / participant / task, ev)
                scores[(participant, task)] = row["joint_rmse_mean"]
                LOGGER.info(f"  {label:<10} {participant}/{task}: "
                            f"{row['joint_rmse_mean']} deg")
            except Exception as exc:
                LOGGER.warning(f"  {label} {participant}/{task} failed: {exc}")
        per_mode[label] = scores

    shared = set(per_mode["confidence"]) & set(per_mode["distance"])
    print(f"\n{'trial':<28}{'confidence':>12}{'distance':>12}{'change':>10}")
    print("-" * 62)
    for key in sorted(shared):
        a, b = per_mode["confidence"][key], per_mode["distance"][key]
        print(f"{'/'.join(key):<28}{a:>12.2f}{b:>12.2f}{b-a:>+10.2f}")
    print("-" * 62)
    a = np.mean([per_mode["confidence"][k] for k in shared])
    b = np.mean([per_mode["distance"][k] for k in shared])
    print(f"{'mean':<28}{a:>12.2f}{b:>12.2f}{b-a:>+10.2f}   deg")
    better = sum(per_mode["distance"][k] < per_mode["confidence"][k] for k in shared)
    print(f"\ndistance weighting is better on {better}/{len(shared)} trials, "
          f"mean change {b-a:+.3f} deg")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as handle:
        handle.write("participant,task,confidence_deg,distance_deg\n")
        for key in sorted(shared):
            handle.write(f"{key[0]},{key[1]},{per_mode['confidence'][key]},"
                         f"{per_mode['distance'][key]}\n")
    print(f"written to {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
