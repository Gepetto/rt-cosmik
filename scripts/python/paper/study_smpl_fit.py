#!/usr/bin/env python3
"""Does fitting a SMPL body to NLF's dense output improve the IK, and at what cost?

Runs the same trials through the plain NLF arm and through several SMPL-fitting
configurations, so the fit is the only thing that changes. Three questions, in
the order they matter:

*Does it help at all?* Compare ``nlf`` against any ``nlfsmpl`` row. If the fit
does not beat the baseline, nothing else here is worth reading.

*How should the shape be handled?* ``free`` refits the body every frame, which
lets the subject change size between frames -- the very artefact the fit exists
to remove. ``calibrated`` measures the subject once over the first second and
then holds it, which is both causal and how a real session would run.
``shared`` fits one shape over the whole trial offline; it cannot ship, and is
here only to bound what the causal modes give up.

*What does it cost?* Reported per frame, unoptimised, alongside the NLF
inference it adds to. The guide's headline is that the cost is per-call
overhead rather than the solve, so ``num_iter`` is nearly free and batching is
what pays -- both testable here.

    python3 scripts/python/paper/study_smpl_fit.py
    python3 scripts/python/paper/study_smpl_fit.py --cameras 0 2 4 6
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

TRIALS = [("1012", "Lifting"), ("1118", "Screwing"), ("4279", "SideOverhead"),
          ("1602", "RobotPolishing"), ("2112", "Polishing"), ("4801", "RobotWelding")]

#: (label, arm, kwargs). The baseline first, so every later row is read against it.
CONFIGS = [
    ("NLF, no fit",            "nlf",     {}),
    ("SMPL, beta per frame",   "nlfsmpl", {"beta_mode": "free"}),
    ("SMPL, beta calibrated",  "nlfsmpl", {"beta_mode": "calibrated"}),
    ("SMPL, calibrated, 1 it", "nlfsmpl", {"beta_mode": "calibrated", "num_iter": 1}),
]

# "shared" is not here: fitting one shape over a whole trial needs two passes,
# which the streaming sweep cannot express, and it could not ship anyway. The
# calibrated mode is the causal version of the same idea.


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/root/workspace/COMFI")
    ap.add_argument("--cameras", type=int, nargs="+", default=[0])
    ap.add_argument("--work-dir", default=str(REPO / "results" / "smpl_study"))
    ap.add_argument("--trials", type=int, default=len(TRIALS),
                    help="how many of the fixed trial list to use")
    args = ap.parse_args()

    import sweep as sweep_mod
    from rtcosmik.config_loader import settings

    ev = sweep_mod._load_eval()
    work = Path(args.work_dir)
    trials = TRIALS[:args.trials]
    cams = "-".join(str(c) for c in args.cameras)

    results = {}
    for label, arm, kwargs in CONFIGS:
        scores, fps, seen = [], [], 0
        for participant, task in trials:
            out = work / f"{label.replace(' ', '_').replace(',', '')}_{cams}"
            out = out / f"{participant}_{task}"
            try:
                frames, _, seconds = sweep_mod.ARMS[arm](
                    args.dataset, participant, task, args.cameras, out, settings,
                    **kwargs)
                # Scored against our own mocap reference -- the same markers
                # through the same model and IK -- not the dataset's published
                # angles, which come from a different biomechanical model.
                scores.append(sweep_mod.score(
                    out, Path(settings.output_dir) / participant / task /
                    "mocap_reference", ev)["joint_rmse_mean"])
                fps.append(frames / seconds)
                seen += 1
            except Exception as exc:
                print(f"  {label} {participant}/{task}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        if seen:
            results[label] = (float(np.mean(scores)), float(np.mean(fps)), scores)
            print(f"{label:<26} {np.mean(scores):6.2f} deg   {np.mean(fps):6.1f} fps   "
                  f"{[round(s, 2) for s in scores]}", flush=True)

    if "NLF, no fit" not in results:
        return 1
    base = results["NLF, no fit"][0]
    print(f"\n{'configuration':<26}{'RMSE':>9}{'vs no fit':>12}{'fps':>9}")
    for label, (mean, rate, _) in results.items():
        delta = "" if label == "NLF, no fit" else f"{mean - base:+.2f}"
        print(f"{label:<26}{mean:>9.2f}{delta:>12}{rate:>9.1f}")
    print("\nNegative means the fit helped. fps is end to end and unoptimised:")
    print("NLF inference, the fit, the filter and the IK, one frame at a time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
