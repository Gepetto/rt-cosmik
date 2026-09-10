#!/usr/bin/env python3
"""Emit the paper's tables as LaTeX and its scaling figure as a PDF.

Reads the summaries scored against the mocap modality plus the cached per-group
metrics, and writes into results/paper/. Nothing here recomputes the study; it
only formats what the sweep produced, so the numbers in the paper and the
numbers in the CSVs cannot drift apart.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
import numpy as np

OUT = REPO / "results" / "paper"
CACHE = Path("/tmp/claude-0/-root-workspace/c71843fd-9421-4298-bc8a-3237ff714e05/scratchpad/cache.json")
MODALITY = {"Base 2-cams": r"2D-MV 2 cams", "Base 4-cams": r"2D-MV 4 cams",
            "Ours 1-cam": r"3D-MV 1 cam", "Ours 2-cams": r"3D-MV 2 cams",
            "Ours 4-cams": r"3D-MV 4 cams"}
ORDER = ["Base 2-cams", "Base 4-cams", "Ours 1-cam", "Ours 2-cams", "Ours 4-cams"]
TASKS = ["Screwing", "Polishing", "SideOverhead", "RobotPolishing",
         "RobotWelding", "Lifting"]
SHORT = {"SideOverhead": "Overhead", "RobotPolishing": "R. polishing",
         "RobotWelding": "R. welding"}


def cell(values):
    per = [np.mean([x[0] for x in v]) for v in values.values()]
    rs = [np.mean([x[1] for x in v]) for v in values.values()]
    return (f"{np.mean(per):.1f} ({np.std(per):.1f})",
            f"{np.nanmean(rs):.2f} ({np.nanstd(rs):.2f})")


def accuracy_table():
    cache = json.load(open(CACHE))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, groups in cache.items():
        ref, label, participant, task = key.split("|")
        if ref != "ourik":
            continue
        for group, value in groups.items():
            data[group][(label, task)][participant].append(value)

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Joint-angle RMSE (deg) and Pearson correlation $r$ against the "
             r"MoCap modality, mean (std) across the 18 participants. All modalities "
             r"share one biomechanical model, one marker convention and one IK; only "
             r"the input differs. 29 degrees of freedom are scored.}",
             r"\label{tab:accuracy}",
             r"\begin{tabular}{ll" + "cc" * len(TASKS) + r"}", r"\toprule",
             r"& & " + " & ".join(r"\multicolumn{2}{c}{%s}" % SHORT.get(t, t)
                                  for t in TASKS) + r" \\",
             r"& & " + " & ".join([r"RMSE & $r$"] * len(TASKS)) + r" \\", r"\midrule"]
    for group, title in (("lower", "Lower limbs"), ("upper", "Upper limbs"),
                         ("trunk", "Trunk"), ("ALL", r"\textbf{Average}")):
        for i, label in enumerate(ORDER):
            row = [title if i == 0 else "", MODALITY[label]]
            for task in TASKS:
                rmse, r = cell(data[group][(label, task)])
                row += [rmse, r]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule" if group != "ALL" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def depth_table(rows):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Marker error split along camera~0's optical axis (depth) and "
             r"perpendicular to it (lateral), in mm, mean (std) over trials. Only the "
             r"four-camera 3D modality reaches an isotropic error, i.e.\ resolves depth "
             r"rather than merely reducing it.}",
             r"\label{tab:depth}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Modality & Depth & Lateral & Depth/lateral \\", r"\midrule"]
    for label, d, ds, l, ls in rows:
        ratio = d / l
        bold = r"\textbf{%.2f}" % ratio if ratio < 1.2 else "%.2f" % ratio
        lines.append(f"{MODALITY[label]} & {d:.1f} ({ds:.1f}) & {l:.1f} ({ls:.1f}) & {bold} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def scaling_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(name):
        return [float(r["joint_rmse_mean"])
                for r in csv.DictReader(open(REPO / "results" / "vs_mocap" / f"{name}.csv"))
                if r["status"] == "ok"]

    series = {"2D-MV": [(2, load("mmpose_0-2")), (4, load("mmpose_0-2-4-6"))],
              "3D-MV": [(1, load("nlf_0")), (2, load("nlf_0-2")), (4, load("nlf_0-2-4-6"))]}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for (name, points), colour, marker in zip(series.items(), ("#1f77b4", "#2ca02c"), ("s", "o")):
        x = [p[0] for p in points]
        mean = [np.mean(p[1]) for p in points]
        sd = [np.std(p[1]) for p in points]
        ax.errorbar(x, mean, yerr=sd, marker=marker, color=colour, label=name,
                    capsize=3, linewidth=1.6, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"])
    ax.set_xlabel("Cameras"); ax.set_ylabel("Joint RMSE (deg)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "camera_scaling.pdf")
    fig.savefig(OUT / "camera_scaling.png", dpi=200)
    return "camera_scaling.pdf"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_accuracy.tex").write_text(accuracy_table() + "\n")
    depth_rows = json.load(open(OUT / "depth_rows.json")) if (OUT / "depth_rows.json").exists() else None
    if depth_rows:
        (OUT / "table_depth.tex").write_text(depth_table(depth_rows) + "\n")
    print("wrote", scaling_figure())
    for f in sorted(OUT.iterdir()):
        print(" ", f.name, f.stat().st_size, "bytes")
