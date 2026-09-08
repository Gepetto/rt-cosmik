#!/usr/bin/env python3
"""Score the arms against the mocap-driven reference instead of COMFI's angles.

COMFI's published joint angles come from COMFI's model and IK, so scoring
against them charges each arm for a model difference it did not cause. Running
the mocap markers through our own model and solver produces a reference on the
same footing, and scoring against that leaves only what the pose estimators did.

Both tables are worth having and they answer different questions:

  vs COMFI      how far the pipeline lands from the dataset's own published
                kinematics -- the number a reader outside this work would expect
  vs our IK     how much of that is the estimator rather than the model

No pipeline is re-run. Both sides already exist on disk as joint_angles.csv; this
only re-scores.

    python3 scripts/python/paper/rescore.py results/mmpose_0-2-4-6.csv ...
"""
import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np


def load_eval():
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_against(run_dir, reference_dir, ev):
    run = ev.load_run(str(run_dir))
    reference = ev.load_run(str(reference_dir))
    lag, _ = ev.estimate_lag(run, reference)
    a, b = ev.apply_lag(run, reference, lag)
    joints, _ = ev.compare_joint_angles(a, b)
    scored = [r["rmse"] for r in joints
              if r["unit"] == "deg" and not r.get("locked")]
    return float(np.mean(scored)) if scored else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--output-root", default=None,
                    help="Where run directories live; default is settings.output_dir")
    ap.add_argument("--reference-tag", default="mocap_reference")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    root = Path(args.output_root or settings.output_dir)
    ev = load_eval()

    per_config = defaultdict(dict)
    for path in args.summaries:
        name = Path(path).stem
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                key = (row["participant"], row["task"])
                run_dir = root / row["participant"] / row["task"] / name
                ref_dir = root / row["participant"] / row["task"] / args.reference_tag
                if not (run_dir.is_dir() and ref_dir.is_dir()):
                    continue
                try:
                    per_config[name][key] = (
                        float(row["joint_rmse_mean"]),
                        score_against(run_dir, ref_dir, ev))
                except Exception as exc:
                    print(f"  {name} {key}: {type(exc).__name__}: {exc}", file=sys.stderr)

    configs = [c for c in sorted(per_config) if per_config[c]]
    if not configs:
        raise SystemExit("no trials had both a run and a mocap reference on disk")
    shared = set.intersection(*(set(per_config[c]) for c in configs))
    print(f"\n{len(shared)} trials scored both ways\n")
    print(f"{'configuration':<22}{'vs COMFI IK':>14}{'vs our IK':>12}{'difference':>13}")
    print("-" * 61)
    for config in configs:
        a = np.mean([per_config[config][k][0] for k in shared])
        b = np.mean([per_config[config][k][1] for k in shared])
        print(f"{config:<22}{a:>14.2f}{b:>12.2f}{b - a:>+13.2f}")
    print("-" * 61 + "   deg")
    print("\nA large drop means the published-angle number was dominated by the\n"
          "difference between the two biomechanical models, not by estimation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
