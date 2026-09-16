#!/usr/bin/env python3
"""Posture-REBA agreement between every arm and the mocap reference, per trial.

For each trial, the arm and the reference are aligned with the same
knee-flexion lag as every other metric, a posture REBA is computed for every
frame of both (``rtcosmik.ergonomics.reba_posture``), and the two are compared:

* REBA score: mean signed and absolute difference, exact agreement (%)
* risk level (negligible / low / medium / high / very high): agreement (%), and
  the time-in-level error -- half the summed absolute difference between the two
  distributions over levels, i.e. the share of the trial spent in a wrong level
* each component (neck, trunk, legs, upper arm, lower arm): agreement (%)
* the frames' risk levels are also written out, so a weighted kappa can be pooled
  per participant downstream rather than computed on one trial

The neutral posture comes from the calibration pose every COMFI trial starts
with (``reba_posture.NEUTRAL_FRAMES``, the first 0.5 s), two ways, both reported:

``reference``  the reference's own start of trial, used for the arm as well. Any
               systematic offset of the arm -- including a different marker
               definition of the head -- counts as error.
``own``        each series' own start of trial. What a deployed system that
               calibrates on the participant's first stance would see: constant
               offsets of the arm cancel.

    python3 scripts/python/paper/reba_agreement.py --arms nlf_0-2-4-6 fastsam_0
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

REFERENCE_TAG = "mocap_reference"
COMPONENTS = ("neck", "trunk", "legs", "upper_arm", "lower_arm")
FIELDS = (["arm", "participant", "task", "neutral", "frames", "reba_arm_mean", "reba_ref_mean",
           "reba_bias", "reba_mae", "reba_exact_pct", "risk_agree_pct", "time_in_level_err_pct"]
          + [f"{c}_agree_pct" for c in COMPONENTS])


def load_eval():
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trials_for(results, arm):
    """Trials an arm completed, from its rescored summary."""
    path = results / "vs_mocap" / f"{arm}.csv"
    if not path.exists():
        return []
    return [(r["participant"], r["task"]) for r in csv.DictReader(open(path))
            if r.get("status") == "ok"]


def compare(ev, run_dir, ref_dir):
    """Rows for both neutral modes, and the per-frame risk levels of each."""
    from rtcosmik.ergonomics import reba_posture as rp
    aj, am = pd.read_csv(run_dir / "joint_angles.csv"), pd.read_csv(run_dir / "markers.csv")
    rj, rm = pd.read_csv(ref_dir / "joint_angles.csv"), pd.read_csv(ref_dir / "markers.csv")
    own = rp.neutral_from(aj.iloc[rp.NEUTRAL_FRAMES], am.iloc[rp.NEUTRAL_FRAMES])
    ref = rp.neutral_from(rj.iloc[rp.NEUTRAL_FRAMES], rm.iloc[rp.NEUTRAL_FRAMES])

    lag, _ = ev.estimate_lag(ev.load_run(str(run_dir)), ev.load_run(str(ref_dir)))
    s0, r0 = max(0, lag), max(0, -lag)
    aj, am, rj, rm = aj.iloc[s0:], am.iloc[s0:], rj.iloc[r0:], rm.iloc[r0:]
    n = min(len(aj), len(rj))
    aj, am, rj, rm = (x.iloc[:n].reset_index(drop=True) for x in (aj, am, rj, rm))

    sr = rp.scores(rj, rm, ref)
    out = {}
    for mode, arm_neutral in (("reference", ref), ("own", own)):
        sa = rp.scores(aj, am, arm_neutral)
        ok = sa["valid"] & sr["valid"]
        ea, er, ka, kr = sa["reba"][ok], sr["reba"][ok], sa["risk"][ok], sr["risk"][ok]
        row = {"frames": int(ok.sum()), "reba_arm_mean": float(ea.mean()),
               "reba_ref_mean": float(er.mean()), "reba_bias": float((ea - er).mean()),
               "reba_mae": float(np.abs(ea - er).mean()),
               "reba_exact_pct": float(100 * np.mean(ea == er)),
               "risk_agree_pct": float(100 * np.mean(ka == kr)),
               "time_in_level_err_pct": float(50 * sum(abs(np.mean(ka == i) - np.mean(kr == i))
                                                       for i in range(len(rp.RISK_NAMES))))}
        row.update({f"{c}_agree_pct": float(100 * np.mean(sa[c][ok] == sr[c][ok]))
                    for c in COMPONENTS})
        out[mode] = (row, np.column_stack([ka, kr]).astype(np.int8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="where run folders live; defaults to settings.output_dir")
    ap.add_argument("--results", type=Path, default=REPO / "results",
                    help="folder whose vs_mocap/ lists the trials each arm completed")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "paper")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = load_eval()
    runs_root = args.output_dir or Path(settings.output_dir)

    for arm in args.arms:
        out_csv = args.out / "reba" / f"{arm}.csv"
        levels_dir = args.out / "reba" / "levels" / arm
        levels_dir.mkdir(parents=True, exist_ok=True)
        trials = trials_for(args.results, arm)
        written, failed = 0, 0
        with open(out_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for participant, task in trials:
                try:
                    modes = compare(ev, runs_root / participant / task / arm,
                                    runs_root / participant / task / REFERENCE_TAG)
                except Exception as exc:
                    failed += 1
                    print(f"  {arm} {participant}/{task}: {type(exc).__name__}: {exc}")
                    continue
                for mode, (row, levels) in modes.items():
                    writer.writerow({"arm": arm, "participant": participant, "task": task,
                                     "neutral": mode, **row})
                    np.save(levels_dir / f"{participant}_{task}_{mode}.npy", levels)
                written += 1
        print(f"{arm}: {written} trials -> {out_csv}" + (f", {failed} failed" if failed else ""),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
