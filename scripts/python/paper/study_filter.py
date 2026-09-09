#!/usr/bin/env python3
"""Does the IIR cutoff earn its current value?

settings ships order 4 at a 10 Hz cutoff, sampling at 40 Hz -- half of Nyquist,
which is permissive. Human motion in these tasks sits well below that, and the
markerless input is noisy (a 15 px 2D error is ~57 mm of ray uncertainty per
camera), so a lower cutoff may trade nothing for real accuracy.

The filter is shared by both modalities, so whatever wins here has to be applied
to both or the comparison stops being an ablation. This sweeps the mmpose arm
because it needs no GPU; a win is then worth confirming on NLF.

    python3 scripts/python/paper/study_filter.py --dataset /root/workspace/COMFI
"""
import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

import sweep as sweep_mod

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s", force=True)
LOGGER = logging.getLogger("filter")
LOGGER.setLevel(logging.INFO)

TRIALS = [("1012", "Lifting"), ("1118", "Screwing"), ("4279", "SideOverhead"),
          ("1602", "RobotPolishing"), ("1847", "Polishing"), ("4801", "RobotWelding"),
          ("2112", "Lifting"), ("4665", "Screwing")]


def run(dataset, participant, task, cameras, out_dir, settings):
    from rtcosmik.paper.mmpose_baseline import build_source
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    source, meta = build_source(dataset, participant, task, cameras, settings)
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(out_dir),
                     markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    for frame, mks in source:
        q = solver.solve(mks)
        row = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            position = mks[name]
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = map(float, position)
        saver.save_markers(row)
        saver.save_joint_angles(
            OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
    saver.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[2, 3, 4, 5, 6, 8, 10, 14])
    ap.add_argument("--orders", type=int, nargs="+", default=[4])
    ap.add_argument("--summary", default="results/filter_study.csv")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = sweep_mod._load_eval()
    base_cutoff, base_order = settings.cutoff_freq, settings.order
    rows = []
    try:
        for order in args.orders:
            for cutoff in args.cutoffs:
                settings.order, settings.cutoff_freq = order, cutoff
                scores, markers = [], []
                for participant, task in TRIALS:
                    out = (Path(settings.output_dir) / participant / task
                           / f"filter_o{order}_c{cutoff:g}")
                    try:
                        run(args.dataset, participant, task, args.cameras, out, settings)
                        r = sweep_mod.score(
                            out, Path(settings.output_dir) / participant / task
                            / "mocap_reference", ev)
                        scores.append(r["joint_rmse_mean"])
                        markers.append(r["marker_mm"])
                    except Exception as exc:
                        LOGGER.warning(f"  o{order} c{cutoff} {participant}/{task}: {exc}")
                if scores:
                    rows.append((order, cutoff, float(np.mean(scores)),
                                 float(np.std(scores)), float(np.mean(markers))))
                    LOGGER.info(f"order {order}, cutoff {cutoff:>4g} Hz: "
                                f"{np.mean(scores):6.3f} deg, {np.mean(markers):6.1f} mm")
    finally:
        settings.order, settings.cutoff_freq = base_order, base_cutoff

    print(f"\n{'order':>6}{'cutoff Hz':>11}{'joint RMSE':>13}{'std':>8}{'marker mm':>12}")
    print("-" * 50)
    for order, cutoff, mean, sd, mm in rows:
        flag = "  <-- shipped" if (order == base_order and cutoff == base_cutoff) else ""
        print(f"{order:>6}{cutoff:>11g}{mean:>13.3f}{sd:>8.3f}{mm:>12.1f}{flag}")
    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"\nbest: order {best[0]}, cutoff {best[1]:g} Hz -> {best[2]:.3f} deg")
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as handle:
        handle.write("order,cutoff_hz,joint_rmse_deg,std_deg,marker_mm\n")
        for row in rows:
            handle.write(",".join(f"{v:g}" for v in row) + "\n")
    print(f"written to {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
