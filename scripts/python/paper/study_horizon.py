#!/usr/bin/env python3
"""Which MHE horizon N gives the best accuracy?

Sweeps settings.N over a subset of trials and reports the mean joint RMSE for
each value. N changes the generated OCP, so every value needs its own artefact;
this builds them as it goes.

    python3 scripts/python/paper/study_horizon.py --dataset /root/workspace/COMFI \
        --n-values 3 5 7 10 15 20

This is the one place N is set from the command line, because varying it is the
entire experiment. settings.py stays the source of truth for real runs -- the
value written there is restored when the study finishes.

Note that N is not only the horizon: run_pipeline and the mmpose driver also use
it as the block length of the IIR filter, so a change moves the smoothing as
well as the estimator. The two effects cannot be separated without changing the
pipeline, so read the result as "best N for the pipeline as built", which is
also what it means for anyone tuning settings.py.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

import sweep as sweep_mod

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s", force=True)
LOGGER = logging.getLogger("horizon")
LOGGER.setLevel(logging.INFO)

DEFAULT_TRIALS = [
    # Spread over subjects and over movement types: a deep squat, a lift, a
    # walk, an overhead reach and a seated task, so the answer is not tuned to
    # one kind of motion.
    ("1012", "Lifting"), ("1012", "Squatting"), ("1012", "StraightWalking"),
    ("1118", "Lifting"), ("1118", "StraightWalking"),
    ("4279", "SideOverhead"), ("4279", "Squatting"),
    ("1508", "SitToStand"), ("1602", "CircularWalking"), ("1847", "Picking"),
]


def regenerate(settings, profile):
    """Build the OCP for the current settings.N."""
    import run_ocp_codegen as codegen
    model, keys = codegen.structural_model()
    if settings.mhe_backend == "acados":
        codegen.generate_acados(model, keys, profile)
    else:
        codegen.generate_fatrop(model, keys, profile)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-values", type=int, nargs="+", default=[3, 5, 7, 10, 15, 20])
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--summary", default="results/horizon.csv")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts" / "python" / "core"))
    from rtcosmik.config_loader import settings
    ev = sweep_mod._load_eval()

    original = settings.N
    profile = settings.mhe_profile
    summary = Path(args.summary)
    per_n = {}

    try:
        for n in args.n_values:
            settings.N = n
            LOGGER.info(f"=== N = {n}: generating {settings.mhe_backend}/{profile} ===")
            t0 = time.perf_counter()
            regenerate(settings, profile)
            LOGGER.info(f"    built in {time.perf_counter()-t0:.0f} s")

            scores, fps = [], []
            for participant, task in DEFAULT_TRIALS:
                out_dir = (Path(settings.output_dir) / participant / task
                           / f"horizon_N{n}")
                try:
                    frames, ik_ms, seconds = sweep_mod.run_mmpose(
                        args.dataset, participant, task, args.cameras,
                        out_dir, settings)
                    row = sweep_mod.score(
                        out_dir,
                        Path(args.dataset) / "mocap" / "aligned" / participant / task,
                        ev)
                    scores.append(row["joint_rmse_mean"])
                    fps.append(frames / seconds)
                    LOGGER.info(f"    {participant}/{task}: {row['joint_rmse_mean']} deg")
                except Exception as exc:
                    LOGGER.warning(f"    {participant}/{task} failed: {exc}")
            per_n[n] = (float(np.mean(scores)) if scores else float("nan"),
                        float(np.median(scores)) if scores else float("nan"),
                        float(np.mean(fps)) if fps else float("nan"), len(scores))
            LOGGER.info(f"=== N = {n}: mean {per_n[n][0]:.3f} deg over "
                        f"{per_n[n][3]} trials, {per_n[n][2]:.1f} fps ===")
    finally:
        settings.N = original
        LOGGER.info(f"settings.N restored to {original}; regenerating its OCP")
        try:
            regenerate(settings, profile)
        except Exception as exc:
            LOGGER.error(f"could not restore the OCP for N={original}: {exc}")

    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as handle:
        handle.write("N,mean_joint_rmse_deg,median_joint_rmse_deg,mean_fps,trials\n")
        print(f"\n{'N':>4}{'mean deg':>11}{'median deg':>12}{'fps':>8}{'trials':>8}")
        for n, (mean, median, f, count) in sorted(per_n.items()):
            handle.write(f"{n},{mean:.4f},{median:.4f},{f:.2f},{count}\n")
            print(f"{n:>4}{mean:>11.3f}{median:>12.3f}{f:>8.1f}{count:>8}")
    best = min(per_n, key=lambda k: per_n[k][0])
    print(f"\nbest N = {best} ({per_n[best][0]:.3f} deg)")
    print(f"written to {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
