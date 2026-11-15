#!/usr/bin/env python3
"""Joint-angle traces of one participant on the six tasks, as one figure.

Two degrees of freedom per task (``paper_figures.TRACE_DOFS``), the reference
against the three front ends with four cameras, without time alignment. The
participant is the one whose advantage of NLF-3D over RTMPose+LSTM on those
degrees of freedom is closest to the median, summed over the tasks.

    python3 scripts/python/paper/trace_grid.py --runs RUNS --out fig_traces_grid.pdf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as pf

ARMS = {"nlf": "nlf_0-2-4-6", "mmpose": "mmpose_0-2-4-6", "nlf2d": "nlf2d_0-2-4-6"}
TITLES = {"Lifting": "Lifting", "SideOverhead": "Overhead", "Screwing": "Screwing",
          "Polishing": "Polishing", "RobotPolishing": "Robot polishing",
          "RobotWelding": "Robot welding"}


def advantage(runs, participant):
    """RTMPose+LSTM error minus NLF-3D error on the traced DoF, summed over tasks (deg)."""
    total = 0.0
    for task, dofs in pf.TRACE_DOFS.items():
        error = {}
        for key in ("nlf", "mmpose"):
            scores = pd.read_csv(runs / participant / task / ARMS[key] / "per_dof.csv")
            error[key] = scores.set_index("dof").causal_rmse_deg[list(dofs)].sum()
        total += error["mmpose"] - error["nlf"]
    return total


def series(runs, participant, task, tag, dofs):
    columns = [f"{d}[rad]" for d in dofs]
    frame = pd.read_csv(runs / participant / task / tag / "joint_angles.csv", usecols=columns)
    return np.degrees(frame[columns].to_numpy(float))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plt = pf.setup()
    people = sorted(p.name for p in args.runs.iterdir() if p.is_dir())
    gains = {p: advantage(args.runs, p) for p in people}
    participant = min(people, key=lambda p: abs(gains[p] - np.median(list(gains.values()))))
    samples = int(pf.TRACE_WINDOW_S * pf.FS)
    time = np.arange(samples) / pf.FS

    fig, axes = plt.subplots(2, 6, figsize=(pf.TWO_COLUMNS, 62 * pf.MM), sharex=True)
    for column, (task, dofs) in enumerate(pf.TRACE_DOFS.items()):
        data = {"reference": series(args.runs, participant, task, pf.REFERENCE_TAG, dofs)}
        data.update({key: series(args.runs, participant, task, tag, dofs)
                     for key, tag in ARMS.items()})
        length = min(len(v) for v in data.values())
        starts = range(0, max(1, length - samples), int(pf.FS / 4))
        start = max(starts, key=lambda s: np.nanstd(data["reference"][s:s + samples, 0]))
        for row, dof in enumerate(dofs):
            ax = axes[row, column]
            drawn = []
            for key in ("nlf2d", "mmpose", "nlf", "reference"):
                y = data[key][start:start + samples, row]
                style = {k: v for k, v in pf.STYLE[key].items() if k != "label"}
                ax.plot(time[:len(y)], y, zorder=3 if key == "reference" else 2, **style)
                drawn.append(y)
            low, high = pf.nice_limits(np.concatenate(drawn))
            pf.extreme_ticks(ax, "y", low, high, decimals=0)
            pf.extreme_ticks(ax, "x", 0.0, pf.TRACE_WINDOW_S, decimals=0, labels=(row == 1))
            ax.set_title(pf.dof_label(dof, lines=1) if row else
                         f"{TITLES[task]}\n{pf.dof_label(dof, lines=1)}", fontsize=5.5, pad=2)
        axes[1, column].set_xlabel("Time (s)")
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.13, top=0.80, hspace=0.32, wspace=0.50)
    pf.legend_above(fig, ["reference", "nlf", "mmpose", "nlf2d"], 0.90)
    fig.savefig(args.out)
    print(f"participant {participant}, advantage {gains[participant]:.1f} deg (median "
          f"{np.median(list(gains.values())):.1f}); written to {args.out}")


if __name__ == "__main__":
    main()
