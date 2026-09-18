#!/usr/bin/env python3
"""The paper's plots, in the style of the first RT-COSMIK paper's figures.

House style (``main/figures/plots_cosmik.pdf``): one horizontal and one vertical
axis per panel, no frame or grid; every axis names its quantity and unit; ticks
only at the two extreme values, same decimals; a shared axis labelled once;
legend on one line above the panels, no frame, once per figure; no title; the
reference a thick black line, the arms in fixed colours and line styles (below),
so they stay apart in grayscale and for colour-blind readers. Widths: 88 mm for
one column, 181 mm for two; fonts 8 pt at that size (Liberation Sans, metric
twin of the Arial used by the original figures).

Every figure is written as ``<name>.pdf`` (vector), ``<name>.png`` (200 dpi
preview) and ``<name>.csv`` (the numbers drawn, to restyle without re-running).

    python3 scripts/python/paper/paper_figures.py traces
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

MM = 1 / 25.4
ONE_COLUMN, TWO_COLUMNS = 88 * MM, 181 * MM
FS = 40.0
REFERENCE_TAG = "mocap_reference"
TRACE_WINDOW_S = 15.0     # seconds of a trial shown in the trace figures

#: Colour and line of each front end, the same in every figure. Checked for
#: colour-vision deficiencies (OKLab distance >= 16 under protan and deutan
#: simulation) and ordered in lightness, so they also separate in grayscale.
STYLE = {
    "reference": dict(color="#000000", lw=1.6, ls="-", label="Reference"),
    "nlf": dict(color="#16782c", lw=0.9, ls="-", label="NLF-3D"),
    "mmpose": dict(color="#1b4fb3", lw=0.9, ls=(0, (4, 1.5)), label="RTMPose+LSTM"),
    "nlf2d": dict(color="#f39a1e", lw=0.9, ls=(0, (5, 1.5, 1, 1.5)), label="NLF-2D"),
    "fastsam": dict(color="#a877cf", lw=0.9, ls=(0, (1, 1.2)), label="FastSAM-3D"),
}
CAMERAS = {"0-2-4-6": "4", "0-2": "2S", "0-4": "2F", "0": "1"}

SAGITTAL = ("Right_Knee_Flexion_Extension", "Left_Knee_Flexion_Extension",
            "Right_Shoulder_Flexion_Extension", "Left_Shoulder_Flexion_Extension")
AXIAL = ("Right_Shoulder_Internal_External_Rotation", "Left_Shoulder_Internal_External_Rotation",
         "Right_Elbow_Pronation_Supination", "Left_Elbow_Pronation_Supination")
DOF_LABELS = {"Knee_Flexion_Extension": ("knee", "flex./ext."),
              "Shoulder_Flexion_Extension": ("shoulder", "flex./ext."),
              "Shoulder_Internal_External_Rotation": ("shoulder", "int./ext. rot."),
              "Elbow_Pronation_Supination": ("elbow", "pron./sup.")}


def dof_label(dof, lines=1):
    """'R. knee flex./ext. (deg)', or on two lines: joint, then motion and unit."""
    side, rest = dof.split("_", 1)
    joint, motion = DOF_LABELS[rest]
    return f"{side[0]}. {joint}" + ("\n" if lines == 2 else " ") + f"{motion} (deg)"


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


# --------------------------------------------------------------------------- style

def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in font_manager.findSystemFonts():      # matplotlib's cache may predate them
        if "Liberation" in path:
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelpad": 2,
        "savefig.dpi": 200, "figure.dpi": 100,
    })
    return plt


def nice_limits(values, step=None):
    """Axis limits rounded outwards to a step that suits the range."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if step is None:
        span = max(hi - lo, 1e-9)
        step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if span / s <= 12)
    return step * np.floor(lo / step), step * np.ceil(hi / step)


def extreme_ticks(ax, axis, lo, hi, decimals=1, labels=True):
    """Limits and ticks at the two extremes only, with the same decimals."""
    ticks = [lo, hi]
    text = [f"{v:.{decimals}f}" for v in ticks] if labels else ["", ""]
    if axis == "x":
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.set_xticklabels(text)
    else:
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels(text)


def legend_above(fig, keys, axes_top):
    """One horizontal legend line above the panels, no frame."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **{k: v for k, v in STYLE[key].items() if k != "label"}) for key in keys]
    fig.legend(handles, [STYLE[k]["label"] for k in keys], loc="lower center",
               bbox_to_anchor=(0.5, axes_top), ncol=len(keys), frameon=False,
               handlelength=2.0, handletextpad=0.5, columnspacing=1.2, borderaxespad=0.0)


def panel_label(ax, text):
    """(a), (b), ... at the panel's top right, just above the plotting area."""
    ax.text(1.0, 1.0, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def save(fig, out, name, header, rows):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    with open(out / f"{name}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {name}: pdf, png, csv -> {out}", flush=True)


# --------------------------------------------------------------------------- traces

def pick_trial(paper, task, arms):
    """The participant whose trial is most typical for both arms: the smallest
    summed distance of the two arms' whole-body RMSE to their task medians."""
    rmse = {}
    for arm in arms:
        rows = {r["participant"]: float(r["joint_rmse_deg"])
                for r in read(paper / "per_trial" / f"{arm}.csv") if r["task"] == task}
        rmse[arm] = rows
    common = sorted(set.intersection(*(set(v) for v in rmse.values())))
    medians = {arm: np.median([rmse[arm][p] for p in common]) for arm in arms}
    return min(common, key=lambda p: sum(abs(rmse[a][p] - medians[a]) / medians[a] for a in arms))


def aligned_series(runs_root, paper, participant, task, arms, dofs):
    """Degrees on the reference's time base (arm row f + lag at reference frame f)."""
    import pandas as pd
    cols = [f"{d}[rad]" for d in dofs]
    ref = np.degrees(pd.read_csv(runs_root / participant / task / REFERENCE_TAG / "joint_angles.csv",
                                 usecols=cols)[cols].to_numpy(float))
    out, lags = {"reference": ref}, {}
    for arm in arms:
        row = [r for r in read(paper / "per_trial" / f"{arm}.csv")
               if r["participant"] == participant and r["task"] == task][0]
        lag = int(row["lag_frames"])
        values = np.degrees(pd.read_csv(runs_root / participant / task / arm / "joint_angles.csv",
                                        usecols=cols)[cols].to_numpy(float))
        shifted = np.full_like(ref, np.nan)
        idx = np.arange(len(ref)) + lag
        ok = (idx >= 0) & (idx < len(values))
        shifted[ok] = values[idx[ok]]
        out[arm], lags[arm] = shifted, lag
    return out, lags


def traces(args, plt):
    """Two stacked panels, one trial: a sagittal DoF on top, an axial one below,
    reference against NLF-3D and RTMPose+LSTM with four cameras."""
    arms = ["nlf_0-2-4-6", "mmpose_0-2-4-6"]
    for task, name in (("Lifting", "traces_lifting"), ("SideOverhead", "traces_overhead"),
                       ("RobotPolishing", "traces_robot_polishing")):
        participant = pick_trial(args.paper, task, arms)
        per_dof = {arm: {r["dof"]: float(r["rmse_deg"]) for r in read(args.paper / "per_dof" / f"{arm}.csv")
                         if r["participant"] == participant and r["task"] == task} for arm in arms}
        series, lags = aligned_series(args.runs, args.paper, participant, task, arms, SAGITTAL + AXIAL)
        ref = series["reference"]
        # Sagittal: the task's main joint, on the side that moves more; axial: where
        # the two arms differ most.
        joint = "Knee" if task == "Lifting" else "Shoulder"
        sagittal = max((i for i, d in enumerate(SAGITTAL) if joint in d), key=lambda i: np.nanstd(ref[:, i]))
        axial = max(range(len(AXIAL)), key=lambda i: abs(per_dof[arms[0]][AXIAL[i]] - per_dof[arms[1]][AXIAL[i]]))
        columns = [sagittal, len(SAGITTAL) + axial]
        dofs = [SAGITTAL[sagittal], AXIAL[axial]]
        # The window of TRACE_WINDOW_S seconds in which the sagittal DoF moves most.
        n = int(TRACE_WINDOW_S * FS)
        valid = np.all([np.isfinite(series[a][:, columns]).all(axis=1) for a in arms], axis=0)
        starts = [s for s in range(0, len(ref) - n, int(FS / 4)) if valid[s:s + n].all()]
        start = max(starts, key=lambda s: np.nanstd(ref[s:s + n, columns[0]]))
        t = np.arange(n) / FS

        fig, axes = plt.subplots(2, 1, figsize=(ONE_COLUMN, 62 * MM), sharex=True)
        for k, (ax, col, dof) in enumerate(zip(axes, columns, dofs)):
            drawn = []
            for key, source in (("reference", "reference"), ("mmpose", arms[1]), ("nlf", arms[0])):
                y = series[source][start:start + n, col]
                style = {k2: v for k2, v in STYLE[key].items() if k2 != "label"}
                ax.plot(t, y, zorder=3 if key == "reference" else 2, **style)
                drawn.append(y)
            lo, hi = nice_limits(np.concatenate(drawn))
            extreme_ticks(ax, "y", lo, hi)
            ax.set_ylabel(dof_label(dof, lines=2))
            extreme_ticks(ax, "x", 0.0, TRACE_WINDOW_S, labels=(k == 1))
            panel_label(ax, "(a)" if k == 0 else "(b)")
        axes[1].set_xlabel("Time (s)")
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.89, hspace=0.25)
        fig.align_ylabels(axes)
        legend_above(fig, ["reference", "nlf", "mmpose"], 0.91)

        header = ["participant", "task", "window_start_frame", "time_s"] + \
                 [f"{who}:{dof}" for dof in dofs for who in ("reference", "NLF-3D 4", "RTMPose+LSTM 4")] + \
                 ["lag_frames NLF-3D 4", "lag_frames RTMPose+LSTM 4"]
        rows = [[participant, task, start, f"{t[i]:.3f}"] +
                [f"{series[src][start + i, c]:.3f}" for c in columns for src in ("reference", arms[0], arms[1])] +
                [lags[arms[0]], lags[arms[1]]] for i in range(n)]
        save(fig, args.out, name, header, rows)
        plt.close(fig)
        print(f"    {participant}/{task}, {TRACE_WINDOW_S:.0f} s from frame {start}: {dofs[0]} "
              f"(RMSE NLF-3D {per_dof[arms[0]][dofs[0]]:.1f}, RTMPose+LSTM {per_dof[arms[1]][dofs[0]]:.1f} deg), "
              f"{dofs[1]} ({per_dof[arms[0]][dofs[1]]:.1f}, {per_dof[arms[1]][dofs[1]]:.1f} deg)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="+", choices=("traces",))
    ap.add_argument("--results", type=Path, default=REPO / "results" / "campaign")
    ap.add_argument("--runs", type=Path, default=REPO / "output" / "campaign")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.paper = args.results / "paper"
    args.out = args.out or args.paper / "figures"
    plt = setup()
    for name in args.figures:
        {"traces": traces}[name](args, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
