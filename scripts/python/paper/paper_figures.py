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

#: Colour and line of each front end, the same in every figure and rendering.
#: Colours from Paul Tol's and Okabe-Ito's colour-blind-safe sets, chosen so
#: that every pair stays >= 14 apart (OKLab x 100) for normal vision and under
#: protan, deutan and tritan simulation; the line style is the second cue, and
#: the proposed pipeline is the only mid-dark colour, so it also stands out in
#: grayscale.
STYLE = {
    "reference": dict(color="#000000", lw=1.6, ls="-", label="Reference"),
    "nlf": dict(color="#117733", lw=1.0, ls="-", label="NLF-3D"),
    "mmpose": dict(color="#E69F00", lw=1.0, ls=(0, (4, 1.5)), label="RTMPose+LSTM"),
    "nlf2d": dict(color="#56B4E9", lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), label="NLF-2D"),
    "fastsam": dict(color="#B0579A", lw=1.0, ls=(0, (1, 1.2)), label="FastSAM-3D"),
}
FOUR_CAMERAS = {"nlf": "nlf_0-2-4-6", "mmpose": "mmpose_0-2-4-6", "nlf2d": "nlf2d_0-2-4-6",
                "fastsam": "fastsam_0-2-4-6"}
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
        step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if span / s <= 12)
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
    """A horizontal legend above the panels, no frame: one line when it fits the
    figure's width, otherwise two."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **{k: v for k, v in STYLE[key].items() if k != "label"}) for key in keys]
    labels = [STYLE[k]["label"] for k in keys]
    options = dict(loc="lower center", bbox_to_anchor=(0.5, axes_top), frameon=False,
                   handlelength=1.8, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    legend = fig.legend(handles, labels, ncol=len(keys), **options)
    fig.canvas.draw()
    if legend.get_window_extent().width > fig.bbox.width:
        legend.remove()
        # matplotlib fills a legend column by column; reorder so it reads by rows.
        ncol = (len(keys) + 1) // 2
        order = [i for c in range(ncol) for i in (c, c + ncol) if i < len(keys)]
        legend = fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=ncol, **options)
    return legend


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

def pick_trial(paper, task, arms, require=()):
    """The participant whose trial is most typical for ``arms``: the smallest
    summed distance of their whole-body RMSE to the task medians, among the
    participants every arm in ``require`` also covers."""
    rmse = {}
    for arm in dict.fromkeys(list(arms) + list(require)):
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
    keys = ["nlf", "mmpose", "nlf2d", "fastsam"]
    arms = [FOUR_CAMERAS[k] for k in keys]
    for task, name in (("Lifting", "traces_lifting"), ("SideOverhead", "traces_overhead"),
                       ("RobotPolishing", "traces_robot_polishing")):
        participant = pick_trial(args.paper, task, arms[:2], require=arms)
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

        fig, axes = plt.subplots(2, 1, figsize=(ONE_COLUMN, 66 * MM), sharex=True)
        for k, (ax, col, dof) in enumerate(zip(axes, columns, dofs)):
            drawn = []
            for key, source in [(k, FOUR_CAMERAS[k]) for k in ("fastsam", "nlf2d", "mmpose", "nlf")] + \
                    [("reference", "reference")]:
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
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.86, hspace=0.25)
        fig.align_ylabels(axes)
        legend_above(fig, ["reference"] + keys, 0.875)

        who = ["Reference"] + [f"{STYLE[k]['label']} 4" for k in keys]
        header = ["participant", "task", "window_start_frame", "time_s"] + \
                 [f"{w}:{dof}" for dof in dofs for w in who] + [f"lag_frames {w}" for w in who[1:]]
        rows = [[participant, task, start, f"{t[i]:.3f}"] +
                [f"{series[src][start + i, c]:.3f}" for c in columns for src in ["reference"] + arms] +
                [lags[a] for a in arms] for i in range(n)]
        save(fig, args.out, name, header, rows)
        plt.close(fig)
        print(f"    {participant}/{task}, {TRACE_WINDOW_S:.0f} s from frame {start}: {dofs[0]} "
              f"(RMSE NLF-3D {per_dof[arms[0]][dofs[0]]:.1f}, RTMPose+LSTM {per_dof[arms[1]][dofs[0]]:.1f} deg), "
              f"{dofs[1]} ({per_dof[arms[0]][dofs[1]]:.1f}, {per_dof[arms[1]][dofs[1]]:.1f} deg)", flush=True)


# --------------------------------------------------------------------------- ergonomics

RISK_BANDS = ((2, 3, "low"), (4, 7, "medium"), (8, 10, "high"), (11, 15, "very high"))
ERGO_WINDOW_S = 30.0
HOLD_MM = 300.0


def reba_series(runs_root, participant, task, arm, lag):
    """Per-frame posture REBA of a run (its own neutral), on the reference time base."""
    import pandas as pd
    from rtcosmik.ergonomics import reba_posture as rp
    run = runs_root / participant / task / arm
    j, m = pd.read_csv(run / "joint_angles.csv"), pd.read_csv(run / "markers.csv")
    reba = rp.scores(j, m, rp.neutral_from(j.iloc[rp.NEUTRAL_FRAMES], m.iloc[rp.NEUTRAL_FRAMES]))["reba"]
    return reba, lag


def ergonomics(args, plt):
    """One co-manipulation trial: REBA over time with the risk levels as bands,
    and the right hand to end-effector distance, reference and the four front
    ends with four cameras."""
    keys = ["nlf", "mmpose", "nlf2d", "fastsam"]
    arms = [FOUR_CAMERAS[k] for k in keys]
    frames_dir = args.paper / "robot_distance" / "frames"
    # The trial most typical for NLF-3D and RTMPose+LSTM in right-hand distance error.
    rmse = {}
    for arm in arms:
        rmse[arm] = {(r["participant"], r["task"]): float(r["rmse_mm"])
                     for r in read(args.paper / "robot_distance" / f"{arm}.csv") if r["measure"] == "right_hand_ee"}
    common = sorted(set.intersection(*(set(v) for v in rmse.values())))
    medians = {a: np.median([rmse[a][t] for t in common]) for a in arms[:2]}
    participant, task = min(common, key=lambda t: sum(abs(rmse[a][t] - medians[a]) / medians[a] for a in arms[:2]))

    import pandas as pd
    from rtcosmik.ergonomics import reba_posture as rp
    ref_dir = args.runs / participant / task / REFERENCE_TAG
    rj, rm = pd.read_csv(ref_dir / "joint_angles.csv"), pd.read_csv(ref_dir / "markers.csv")
    reba = {"reference": rp.scores(rj, rm, rp.neutral_from(rj.iloc[rp.NEUTRAL_FRAMES], rm.iloc[rp.NEUTRAL_FRAMES]))["reba"]}
    n_ref = len(reba["reference"])
    distance = {}
    for key, arm in zip(keys, arms):
        lag = [int(r["lag_frames"]) for r in read(args.paper / "per_trial" / f"{arm}.csv")
               if r["participant"] == participant and r["task"] == task][0]
        values, _ = reba_series(args.runs, participant, task, arm, lag)
        shifted = np.full(n_ref, np.nan)
        idx = np.arange(n_ref) + lag
        ok = (idx >= 0) & (idx < len(values))
        shifted[ok] = values[idx[ok]]
        reba[key] = shifted
        d = np.load(frames_dir / arm / f"{participant}_{task}.npz")
        dist_arm = np.full(n_ref, np.nan)
        dist_arm[d["ref_frame"]] = 1000 * d["right_hand_ee_arm"]
        distance[key] = dist_arm
        if "reference" not in distance:
            distance["reference"] = np.full(n_ref, np.nan)
            distance["reference"][d["ref_frame"]] = 1000 * d["right_hand_ee_ref"]

    # A window of the co-manipulation itself -- the reference hand within HOLD_MM
    # of the end effector throughout, robot states on every frame -- where the
    # reference REBA changes most.
    n = int(ERGO_WINDOW_S * FS)
    near = np.isfinite(distance["reference"]) & (distance["reference"] < HOLD_MM)
    starts = [s for s in range(0, n_ref - n, int(FS / 4)) if near[s:s + n].all()]
    start = max(starts, key=lambda s: np.nanstd(reba["reference"][s:s + n]))
    t = np.arange(n) / FS
    window = slice(start, start + n)

    fig, axes = plt.subplots(2, 1, figsize=(ONE_COLUMN, 66 * MM), sharex=True)
    ax = axes[0]
    top = int(np.nanmax(np.concatenate([reba[k][window] for k in reba]))) + 1
    lo_y = 1
    for i, (lo, hi, name) in enumerate(RISK_BANDS):
        if lo > top:
            break
        ax.axhspan(lo - 0.5, min(hi, top) + 0.5, color=str(0.95 - 0.05 * i), lw=0, zorder=0)
        ax.text(1.01, (lo - 0.5 + min(hi, top) + 0.5) / 2, name, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color="0.35")
    for key in ["fastsam", "nlf2d", "mmpose", "nlf", "reference"]:
        style = {k: v for k, v in STYLE[key].items() if k != "label"}
        ax.step(t, reba[key][window], where="post", zorder=3 if key == "reference" else 2, **style)
    extreme_ticks(ax, "y", lo_y - 0.5, top + 0.5, decimals=1)
    ax.set_yticks([lo_y, top])
    ax.set_yticklabels([f"{lo_y:d}", f"{top:d}"])
    ax.set_ylim(lo_y - 0.5, top + 0.5)
    ax.set_ylabel("REBA score\n(point)")
    extreme_ticks(ax, "x", 0.0, ERGO_WINDOW_S, labels=False)
    panel_label(ax, "(a)")

    ax = axes[1]
    for key in ["fastsam", "nlf2d", "mmpose", "nlf", "reference"]:
        style = {k: v for k, v in STYLE[key].items() if k != "label"}
        ax.plot(t, distance[key][window], zorder=3 if key == "reference" else 2, **style)
    lo, hi = nice_limits(np.concatenate([distance[k][window] for k in distance]))
    extreme_ticks(ax, "y", max(lo, 0.0), hi, decimals=1)
    ax.set_ylabel("R. hand to end\neffector (mm)")
    extreme_ticks(ax, "x", 0.0, ERGO_WINDOW_S)
    ax.set_xlabel("Time (s)")
    panel_label(ax, "(b)")
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.12, top=0.86, hspace=0.25)
    fig.align_ylabels(axes)
    legend_above(fig, ["reference"] + keys, 0.875)

    who = ["Reference"] + [f"{STYLE[k]['label']} 4" for k in keys]
    header = ["participant", "task", "window_start_frame", "time_s"] + \
             [f"{w}:REBA" for w in who] + [f"{w}:right_hand_ee_mm" for w in who]
    rows = [[participant, task, start, f"{t[i]:.3f}"] +
            [f"{reba[k][start + i]:.0f}" if np.isfinite(reba[k][start + i]) else "" for k in ["reference"] + keys] +
            [f"{distance[k][start + i]:.1f}" if np.isfinite(distance[k][start + i]) else "" for k in ["reference"] + keys]
            for i in range(n)]
    save(fig, args.out, "ergonomics_robot_distance", header, rows)
    plt.close(fig)
    print(f"    {participant}/{task}, {ERGO_WINDOW_S:.0f} s from frame {start}; right-hand RMSE "
          + ", ".join(f"{STYLE[k]['label']} {rmse[a][(participant, task)]:.0f} mm" for k, a in zip(keys, arms)),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="+", choices=("traces", "ergonomics"))
    ap.add_argument("--results", type=Path, default=REPO / "results" / "campaign")
    ap.add_argument("--runs", type=Path, default=REPO / "output" / "campaign")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.paper = args.results / "paper"
    args.out = args.out or args.paper / "figures"
    plt = setup()
    for name in args.figures:
        {"traces": traces, "ergonomics": ergonomics}[name](args, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
