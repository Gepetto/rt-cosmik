#!/usr/bin/env python3
"""Assemble everything the paper needs from one campaign into a single folder.

    python3 scripts/python/paper/build_handoff.py --campaign campaign

Reads ``results/<campaign>/`` and ``output/<campaign>/`` (written by
``run_campaign.sh``) and writes ``results/<campaign>/handoff/``, which is meant
to be copied as is to wherever the paper is written:

    README.md       what was run and how: protocol, definitions, exclusions,
                    decisions that differ from RESULTS_BRIEF.md, caveats
    CHECKLIST.md    every deliverable RESULTS_BRIEF.md asks for, and where it is
                    -- or why it is not there
    summary.json    every aggregated number, machine-readable
    tables/         each table as .md, .csv and .tex (booktabs)
    figures/        vector PDFs (+ PNG previews), one colour per arm throughout
    data/           per-trial and per-DoF tables, REBA and robot-distance rows,
                    timing, sweep summaries, and the joint-angle traces behind
                    the trace figure
    logs/           the campaign's logs, its environment record, and code.diff: the
                    uncommitted changes on top of the recorded commit, if any

Per-frame arrays (REBA levels, robot distances) and the run folders themselves
are left out; they are large and every number in the tables is already
aggregated from the per-trial rows included here.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

import numpy as np

ARMS = ["mmpose_0-2", "mmpose_0-4", "mmpose_0-2-4-6", "nlf2d_0-2", "nlf2d_0-4", "nlf2d_0-2-4-6",
        "nlf_0", "nlf_0-2", "nlf_0-4", "nlf_0-2-4-6",
        "fastsam_0", "fastsam_0-2", "fastsam_0-4", "fastsam_0-2-4-6"]
LABELS = {
    "mmpose_0-2": "mmpose+LSTM, 2 cams same side", "mmpose_0-4": "mmpose+LSTM, 2 cams opposed",
    "mmpose_0-2-4-6": "mmpose+LSTM, 4 cams",
    "nlf2d_0-2": "NLF-2D tri, 2 cams same side", "nlf2d_0-4": "NLF-2D tri, 2 cams opposed",
    "nlf2d_0-2-4-6": "NLF-2D tri, 4 cams",
    "nlf_0": "NLF-3D, 1 cam", "nlf_0-2": "NLF-3D, 2 cams same side", "nlf_0-4": "NLF-3D, 2 cams opposed",
    "nlf_0-2-4-6": "NLF-3D, 4 cams",
    "fastsam_0": "FastSAM-3D, 1 cam", "fastsam_0-2": "FastSAM-3D, 2 cams same side",
    "fastsam_0-4": "FastSAM-3D, 2 cams opposed", "fastsam_0-2-4-6": "FastSAM-3D, 4 cams",
}
#: One colour per arm family, darker with more cameras; used by every figure.
COLOURS = {
    "mmpose_0-2": "#f4a582", "mmpose_0-4": "#e7735a", "mmpose_0-2-4-6": "#ca0020",
    "nlf2d_0-2": "#92c5de", "nlf2d_0-4": "#4a9ac6", "nlf2d_0-2-4-6": "#0571b0",
    "nlf_0": "#c2e699", "nlf_0-2": "#78c679", "nlf_0-4": "#41ab5d", "nlf_0-2-4-6": "#238443",
    "fastsam_0": "#dadaeb", "fastsam_0-2": "#9e9ac8", "fastsam_0-4": "#7b6fb3",
    "fastsam_0-2-4-6": "#54278f",
}
#: Sweep-summary columns kept in the handoff. The accuracy columns there are the
#: sweep's quick scores (free-flyer as a mean of per-axis RMSE, markers as mean
#: magnitude) and differ from the tables' definitions, so they are left out.
SWEEP_COLUMNS = ("arm", "participant", "task", "cameras", "n_horizon", "frames", "ik_ms_median",
                 "ik_ms_p95", "fps", "status")
GROUPS = ("lower", "trunk", "upper")
CAP_MM = 200.0      # marker-error axis limit; points beyond it are named on the figure
TRACE_DOFS = ("Right_Knee_Flexion_Extension[rad]", "Right_Shoulder_Flexion_Extension[rad]")


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def copy_tree(src, dst, pattern="*"):
    n = 0
    for f in sorted(Path(src).glob(pattern)):
        if f.is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


# --------------------------------------------------------------------------- tables

def latex_escape(text):
    out = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("|", r"$|$"), (">=", r"$\geq$"), ("<", r"$<$"), ("+-", r"$\pm$")):
        out = out.replace(a, b)
    return out


def csv_to_tex(path, caption):
    rows = list(csv.reader(open(path)))
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + latex_escape(caption) + "}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
             r"\toprule", " & ".join(latex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(latex_escape(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures

def participant_group_means(per_dof_rows, group):
    per_trial = {}
    for r in per_dof_rows:
        if r["group"] == group:
            per_trial.setdefault((r["participant"], r["task"]), []).append(float(r["rmse_deg"]))
    per_participant = {}
    for (p, _), v in per_trial.items():
        per_participant.setdefault(p, []).append(np.mean(v))
    return {p: float(np.mean(v)) for p, v in per_participant.items()}


def figures(paper, out, arms, runs_root, trace_trial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    # 1. Joint RMSE per DoF group and arm, one dot per participant.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, group in zip(axes, GROUPS):
        for i, arm in enumerate(arms):
            values = list(participant_group_means(read(paper / "per_dof" / f"{arm}.csv"), group).values())
            if not values:
                continue
            ax.bar(i, np.mean(values), color=COLOURS[arm], width=0.75)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(values)), values, s=4, color="k", zorder=3)
        ax.set_title(group)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([LABELS[a] for a in arms], rotation=60, ha="right")
    axes[0].set_ylabel("joint RMSE (deg)")
    save(fig, "joint_rmse_by_group")

    # 2. Depth vs lateral marker error against camera count.
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    stars = []
    families = {"mmpose": "mmpose+LSTM", "nlf2d": "NLF-2D tri", "nlf": "NLF-3D", "fastsam": "FastSAM-3D"}
    for family, label in families.items():
        for field, style in (("marker_depth_mm", "-"), ("marker_lateral_mm", "--")):
            xs, ys = [], []
            for arm in arms:
                if arm.split("_")[0] != family or arm.endswith("_0-4"):
                    continue
                rows = read(paper / "per_trial" / f"{arm}.csv")
                per = {}
                for r in rows:
                    per.setdefault(r["participant"], []).append(float(r[field]))
                if per:
                    xs.append(len(arm.split("_")[1].split("-")))
                    ys.append(np.mean([np.mean(v) for v in per.values()]))
            if xs:
                order = np.argsort(xs)
                colour = COLOURS[[a for a in arms if a.split("_")[0] == family][-1]]
                ax.plot(np.array(xs)[order], np.array(ys)[order], style, marker="o", ms=3, color=colour,
                        label=f"{label}, {'depth' if 'depth' in field else 'lateral'}")
                opposed = read(paper / "per_trial" / f"{family}_0-4.csv")
                if opposed:
                    per = {}
                    for r in opposed:
                        per.setdefault(r["participant"], []).append(float(r[field]))
                    star = np.mean([np.mean(v) for v in per.values()])
                    stars.append((star, label, "depth" if "depth" in field else "lateral"))
                    ax.plot([2], [min(star, CAP_MM)], marker="*", ms=6, color=colour, ls="none")
    off_scale = [(v, l, f) for v, l, f in stars if v > CAP_MM]
    if off_scale:
        ax.set_ylim(0, CAP_MM * 1.05)
        ax.text(2.08, CAP_MM * 0.99, "off scale: " + "; ".join(f"{l} {f} {v:.0f} mm" for v, l, f in off_scale),
                fontsize=5, va="top")
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("cameras (line: same-side pair; star: opposed pair)")
    ax.set_ylabel("marker error RMS (mm)")
    ax.legend(fontsize=5, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "marker_depth_lateral_vs_cameras")

    # 3. Rate per arm against the 30 Hz real-time line.
    summary = json.loads((paper / "summary.json").read_text())
    rates = [(arm, summary["timing"].get(arm, {}).get("estimated_rate_hz")) for arm in arms]
    rates = [(a, r) for a, r in rates if r]
    if rates:
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        ax.bar(range(len(rates)), [r for _, r in rates], color=[COLOURS[a] for a, _ in rates])
        ax.axhline(30, color="k", lw=0.6, ls="--")
        ax.text(len(rates) - 0.5, 32, "30 Hz", ha="right", fontsize=6)
        ax.set_yscale("log")
        ax.set_ylabel("estimated rate (Hz)")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([LABELS[a] for a, _ in rates], rotation=60, ha="right")
        save(fig, "rate_per_arm")

    # 4. Knee and shoulder flexion traces for one trial.
    trace = traces(paper, runs_root, arms, trace_trial)
    if trace is not None:
        data, lags = trace
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True)
        best = min((a for a in arms if a in data),
                   key=lambda a: summary["joint"].get(a, {}).get("all", {}).get("rmse_deg", {}).get("mean") or 1e9)
        shown = [a for a in dict.fromkeys(("mmpose_0-2-4-6", "nlf_0-2-4-6", best)) if a in data]
        for ax, dof in zip(axes, TRACE_DOFS):
            ref = np.degrees(data["mocap_reference"][dof].to_numpy(float))
            ax.plot(np.arange(len(ref)) / 40.0, ref, color="k", lw=0.8, label="mocap reference")
            for arm in shown:
                y = np.degrees(data[arm][dof].to_numpy(float))
                ax.plot(np.arange(len(y)) / 40.0, y, color=COLOURS[arm], lw=0.8, label=LABELS[arm])
            ax.set_ylabel(dof.replace("_Flexion_Extension[rad]", "").replace("_", " ") + " flex. (deg)")
        axes[-1].set_xlabel("time (s)")
        axes[0].legend(fontsize=5, frameon=False)
        save(fig, "trace_" + trace_trial.replace("/", "_"))

    # 5-6. REBA risk-level agreement and robot body-distance error, one dot per participant.
    for name, source, field, where, ylabel in (
            ("reba_risk_agreement", "reba", "risk_agree_pct", ("neutral", "own"), "REBA risk-level agreement (%)"),
            ("robot_body_distance_rmse", "robot_distance", "rmse_mm", ("measure", "body"),
             "body-robot distance RMSE (mm)")):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        shown = 0
        for i, arm in enumerate(arms):
            rows = [r for r in read(paper / source / f"{arm}.csv") if r[where[0]] == where[1]]
            per = {}
            for r in rows:
                per.setdefault(r["participant"], []).append(float(r[field]))
            values = [np.mean(v) for v in per.values()]
            if values:
                ax.bar(i, np.mean(values), color=COLOURS[arm], width=0.75)
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(values)), values, s=4, color="k", zorder=3)
                shown += 1
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([LABELS[a] for a in arms], rotation=60, ha="right")
        ax.set_ylabel(ylabel)
        if shown:
            save(fig, name)
        else:
            plt.close(fig)
    # 7-8. Design studies: horizon (accuracy and solve time vs N), filter (accuracy vs delay).
    study = {f.stem: read(f) for f in sorted((paper / "studies").glob("*.csv"))}
    study = {k: v for k, v in study.items() if v}

    def participant_mean(rows, field):
        per = {}
        for r in rows:
            if r[field] not in ("", "nan"):
                per.setdefault(r["participant"], []).append(float(r[field]))
        values = [np.mean(v) for v in per.values()]
        return (np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0) if values else (np.nan, np.nan)

    horizon = sorted(((int(rows[0]["N"]), v) for v, rows in study.items()
                      if rows[0]["ik_type"] == "mhe" and rows[0]["filter"] == "order 4, 5 Hz"))
    if len(horizon) > 1:
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        n = [h for h, _ in horizon]
        rmse = np.array([participant_mean(study[v], "joint_rmse_deg") for _, v in horizon])
        ax.errorbar(n, rmse[:, 0], yerr=rmse[:, 1], color=COLOURS["nlf_0-2-4-6"], marker="o", ms=3, capsize=2)
        ax.set_xlabel("MHE horizon N")
        ax.set_ylabel("joint RMSE (deg)", color=COLOURS["nlf_0-2-4-6"])
        twin = ax.twinx()
        solve = np.array([participant_mean(study[v], "solve_ms_p50") for _, v in horizon])
        twin.plot(n, solve[:, 0], color="0.4", marker="s", ms=3, ls="--")
        twin.set_ylabel("IK solve time p50 (ms)", color="0.4")
        save(fig, "study_horizon")
    filters = [(v, rows) for v, rows in study.items() if rows[0]["ik_type"] == "mhe" and rows[0]["N"] == "7"]
    if len(filters) > 1:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        for v, rows in filters:
            delay = float(rows[0]["filter_delay_1hz_ms"])
            for field, marker in (("joint_rmse_deg", "o"), ("joint_rmse_zero_lag_deg", "^")):
                m, _ = participant_mean(rows, field)
                ax.scatter(delay, m, marker=marker, s=12,
                           color=COLOURS["nlf_0-2-4-6"] if marker == "o" else "0.4")
            # Label the zero-lag points: they spread with delay, the others overlap.
            ax.annotate(rows[0]["filter"].replace("order ", "o"),
                        (delay, participant_mean(rows, "joint_rmse_zero_lag_deg")[0]),
                        fontsize=5, xytext=(4, -3), textcoords="offset points")
        ax.scatter([], [], marker="o", color=COLOURS["nlf_0-2-4-6"], label="lag-compensated")
        ax.scatter([], [], marker="^", color="0.4", label="zero lag")
        ax.legend(fontsize=5, frameon=False)
        ax.set_xlabel("filter group delay at 1 Hz (ms)")
        ax.set_ylabel("joint RMSE (deg)")
        save(fig, "study_filter")
    return made


def traces(paper, runs_root, arms, trial):
    """Lag-aligned knee and shoulder flexion of every arm for one trial, and the lags."""
    import pandas as pd
    participant, task = trial.split("/")
    ref_path = runs_root / participant / task / "mocap_reference" / "joint_angles.csv"
    if not ref_path.exists():
        return None
    data = {"mocap_reference": pd.read_csv(ref_path, usecols=list(TRACE_DOFS))}
    lags = {}
    for arm in arms:
        path = runs_root / participant / task / arm / "joint_angles.csv"
        row = [r for r in read(paper / "per_trial" / f"{arm}.csv")
               if r["participant"] == participant and r["task"] == task]
        if path.exists() and row:
            lags[arm] = int(row[0]["lag_frames"])
            data[arm] = pd.read_csv(path, usecols=list(TRACE_DOFS))
    # Shift every arm onto the reference's time base, as the metrics do.
    aligned = {"mocap_reference": data["mocap_reference"]}
    for arm, lag in lags.items():
        s0, r0 = max(0, lag), max(0, -lag)
        n = min(len(data[arm]) - s0, len(data["mocap_reference"]) - r0)
        frame = data[arm].iloc[s0:s0 + n].reset_index(drop=True)
        aligned[arm] = pd.concat([pd.DataFrame(np.nan, index=range(r0), columns=frame.columns), frame],
                                 ignore_index=True)
    return aligned, lags


# --------------------------------------------------------------------------- docs

def shell(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, cwd=REPO,
                              timeout=60).stdout.strip()
    except Exception as exc:                     # environment details are best effort
        return f"unavailable ({exc})"


def fmt(stat, digits=2):
    if not stat or stat.get("mean") is None:
        return "--"
    return f"{stat['mean']:.{digits}f} ({stat['sd']:.{digits}f})"


def readme(campaign, summary, coverage_rows, figures_made, commit, dirty):
    joint = summary.get("joint", {})
    headline = "\n".join(
        f"| {LABELS.get(a, a)} | {fmt(joint[a]['all']['rmse_deg'])} | {fmt(joint[a]['lower']['rmse_deg'])} | "
        f"{fmt(joint[a]['trunk']['rmse_deg'])} | {fmt(joint[a]['upper']['rmse_deg'])} |"
        for a in ARMS if a in joint)
    friedman = summary.get("stats", {}).get("joint_rmse", {}).get("friedman") or {}
    friedman_line = (f"Friedman on whole-body RMSE over the {len(friedman['arms'])} real-time arms, "
                     f"{friedman['n']} participants: chi2({friedman['df']}) = {friedman['chi2']:.1f}, "
                     f"p = {friedman['p']:.2e}."
                     if friedman.get("chi2") is not None
                     else "Friedman test not computed (fewer than 5 common participants).")
    coverage = "\n".join(f"| {LABELS.get(a, a)} | {t} | {p} |" for a, t, p in coverage_rows)
    return f"""# RT-COSMIK results handoff: `{campaign}`

Built {datetime.now():%Y-%m-%d %H:%M} from branch `paper/mmpose-baseline`, commit `{commit}`
{"**with uncommitted changes** (see logs/environment.txt)" if dirty else "(clean tree)"}.
Every number here is aggregated by `aggregate_results.py` from the per-trial rows in
`data/`; nothing was edited by hand.

## Headline: whole-body joint RMSE (deg), mean (SD) across participants

| Arm | whole body | lower | trunk | upper |
|---|---|---|---|---|
{headline}

{friedman_line} Planned comparisons: `tables/stats_joint_rmse.md`.

## Coverage

| Arm | trials | participants |
|---|---|---|
{coverage}

## What was compared

Every arm feeds markers to the **same** model, marker set, low-pass filter and
moving-horizon IK; only the source of the markers changes.

| Arm | Markers from |
|---|---|
| mmpose+LSTM | COMFI's precomputed RTMPose (Halpe26) 2D keypoints, confidence-weighted DLT, stock OpenCap v0.3 LSTM augmenter (causal 30-frame window) |
| NLF-2D tri | NLF's 2D keypoints at the marker vertices, uncertainty-weighted DLT |
| NLF-3D (proposed) | NLF's metric 3D at the marker vertices, per view, fused by inverse variance |
| FastSAM-3D (offline) | FastSAM-3D-Body (MHR mesh) exported offline per camera, fused by plain mean |

Camera configurations: 4 cameras; 2 cameras **same side** (0-2: one support,
0.82 m apart, 24 deg between optical axes); 2 cameras **opposed** (0-4: facing
each other across the workspace, 163 deg); and 1 camera for the 3D arms.

NLF runs per image, so the 4-camera NLF-3D sweep recorded every camera's NLF
output, and NLF-3D 1 / 2 / 2-opposed and NLF-2D 4 / 2 / 2-opposed were
**replayed** from that recording through the same filter and IK
(`replay_nlf_arms.py`). Every camera configuration therefore sees identical
NLF output. Replay was checked against direct runs on participant 1012:
NLF-2D 4 cams identical, other configurations 0.3-0.4 mm median marker
difference and at most 0.12 deg of joint RMSE per trial -- with one exception.
**2D triangulation from the opposed pair is ill-conditioned**: the two cameras
face each other (163 deg), so their rays are nearly parallel and depth along the
line between them is barely constrained; in addition, the 2D detector swaps
left and right in back views on some trials (1012/Screwing: the right knee is
labelled on the left knee in 88-89% of frames in cameras 4 and 6). mmpose+LSTM
and NLF-2D both collapse with that pair, and small input differences change the
outcome: NLF-2D opposed scored 27.7 deg directly and 30.6 deg replayed on 1012
(one trial differing by 17 deg), and the acados QP reported minimum-step
warnings only on that configuration. Treat its per-trial numbers as unstable;
the conclusion (2D triangulation fails with facing cameras, 3D fusion does not)
holds either way. Direct runs of every replayed configuration on 3 participants
are in `data/sweep_summaries/timing_run_*.csv` for comparison.

* **Marker set `parity`**: the 35 markers the LSTM baseline can produce; 7 DoF it
  cannot observe are locked (thoracic Z/X/Y, both wrists Z/X), 29 DoF are scored.
* **Filter**: 4th-order Butterworth, 5 Hz, causal, fed one frame at a time.
  Group delay 79 / 81 / 101 ms at 0.5 / 1 / 3 Hz.
* **IK**: moving-horizon estimation, N = 7, acados real-time profile.
* **NLF marker vertices**: fitted to mocap (same method as FastSAM's marker map),
  except the pelvis, hands and face, which stay hand-picked. The fit does not
  overfit: fitted on 14 participants, the map scores 34.0 mm median marker error
  on them and 34.0 mm on the 3 held out (hand-picked vertices: 43.1 / 40.5 mm).
  The fitted pelvis markers tilted the pelvis frame about 8 deg and caused
  shoulder flips; keeping the hand-picked pelvis was better or equal in every
  arm tested.
* **FastSAM marker vertices**: the colleague's 17-subject map; TV8/TV12 dropped;
  Head reconstructed from the ears and nose (sensitivity at most 0.18 deg).

## Evaluation protocol

* **Reference**: COMFI's mocap markers driven through the same model and IK
  (`mocap_reference`). COMFI's published joint angles come from another model
  and are **not used anywhere**.
* **Time alignment**: each run is aligned to the reference by knee-flexion
  cross-correlation before scoring. COMFI's video trails its mocap by about 3
  frames (75 ms), a recording offset, so scoring at zero lag would charge every
  arm for it. The residual lag per arm is reported (`tables/freeflyer.md`, frames
  at 40 Hz) and includes both that offset and the filter delay.
* **Aggregation**: per trial -> mean per participant -> mean (SD, ddof = 1)
  across participants. Bias is reported as mean |per-DoF bias|; SD is the
  offset-removed error, RMSE^2 = bias^2 + SD^2 per DoF.
* **Markers**: RMS error against the raw Vicon markers (29 body markers; facial
  markers have no Vicon counterpart), split exactly two ways: depth (along camera
  0's optical axis) / lateral, and whole-body translation / shape.
* **Statistics**: one value per participant. Friedman across the real-time arms
  (FastSAM excluded), then a small family of **planned comparisons**, Holm-
  corrected: NLF-3D 4 cams vs mmpose+LSTM 4 cams; NLF-3D vs NLF-2D tri at 4 cams;
  NLF-3D 1 vs 2 (same side), 2 vs 4, 1 vs 4; NLF-3D 2 cams same side vs opposed.
  Wilcoxon signed-rank on the participants both arms cover, rank-biserial effect
  size, bootstrap 95% CI (10 000 resamples) of the paired difference. FastSAM
  runs offline (below 3 Hz), so it is outside the family: its comparisons with
  NLF-3D at matched camera counts are reported uncorrected.
* **Shoulder flip rate**: share of frames in which any shoulder DoF is more than
  90 deg from the reference. With the arms overhead, shoulder flexion reaches the
  model's +-180 deg limit (kept on purpose: it follows human range of motion) and
  the IK can settle into the equivalent abducted and rotated configuration. It
  concentrates in SideOverhead.

## Ergonomics: posture REBA (`tables/reba_*.md`, `data/reba/`)

Posture-only REBA (Hignett & McAtamney 2000) per frame, for each arm and for the
reference: load, coupling and activity scores are 0 and wrists score 1 (the
wrists are locked). Trunk flexion is the inclination of the pelvis-to-C7 axis
from vertical. Angles are taken relative to a neutral posture: the median of the
first 0.5 s of each trial, where every participant stands in the calibration
pose. `reference`: the reference's neutral used for the arm too (offsets count
as error). `own`: each series' own neutral (what a deployed system would do).
Reported: REBA MAE and bias, exact agreement, risk-level agreement,
linear-weighted Cohen's kappa on risk levels (pooled per participant),
time-in-level error, and per-component agreement.

## Human-robot distance (`tables/robot_*.md`, `data/robot_distance/`)

RobotPolishing and RobotWelding, Franka Panda, COMFI joint states matched to
video frames by camera timestamp, robot base pose from COMFI. The participant
hand-guides the robot, so:

* `whole`: whole-body minimum distance (mostly 0: contact is the task)
* `body`: minimum distance excluding forearms and hands (usually the head)
* `left_hand_ee`, `right_hand_ee`: hand centre to the Panda's hand frame

The human is the joint-centre skeleton of each run's own scaled model (plus
hands of 0.108 H past the wrists and the head up to stature), distances are to
the Panda's collision meshes. Per trial: bias, MAE, RMSE, SD, r, error on the
closest approach, agreement below 100 / 200 / 300 mm and on contact (< 10 mm),
closest-segment agreement. The system is not safety-rated; this measures how
well each arm localises the operator relative to the robot.

**Separation margin** (ISO/TS 15066 speed-and-separation monitoring): the 95th
and 99th percentiles of (estimate - reference), i.e. how much each arm
overstates the separation, plus a latency term: (filter group delay at 1 Hz +
one frame of processing at the arm's rate) x the reference's 95th-percentile
closing speed. Camera exposure and USB transport are not included. Threshold
agreement is kept in the tables but is noisy near each threshold. Robot states are
missing for 1602/RobotWelding, 1847/RobotPolishing, 2307/RobotWelding.

## Timing (`tables/timing.md`, `data/timing/`)

Rough, meant to separate what runs at 30 Hz or more from what can only run
offline -- not a latency benchmark. RTX 4500 Ada + i9-14900K, arms run one at a
time on a free GPU.

* NLF arms: offline end-to-end throughput from video files (decode, detection,
  NLF, reconstruction, filter, IK). The replayed configurations have no
  throughput of their own: they were also run directly on 3 participants for
  timing (`data/sweep_summaries/timing_run_*.csv`).
* CPU governor set to `performance` for the campaign (see logs/environment.txt).
* mmpose arms: RTMPose 2D inference from the RT-COSMIK draft's Table I (7.1 ms
  for 2 cams, 13.2 ms for 4, same machine) plus the measured rest of the chain.
* FastSAM arms: logged inference time per view on the cluster that produced the
  export (about 390 ms, 2.6 images/s; not the machine above) times the number of
  views, plus the IK.

## Design studies: E3 IK type, E4 MHE horizon, E5 filter (`tables/study_*.md`, `data/studies/`)

On the proposed pipeline (NLF-3D, 4 cameras), every trial. The sweep saved each
frame's per-camera NLF output, and each variant replays it through the same
reconstruction, filter and IK; only the setting under study changes. The
baseline replay (MHE, N = 7, 4th order at 5 Hz) reproduces the sweep's own run
(`logs/studies.log`). Reported per variant: RMSE (lag-compensated and at zero
lag), difference to the baseline with a Wilcoxon p-value, residual lag, the
filter's analytic group delay at 1 Hz, jitter (median |frame-to-frame change|),
RMS jerk, share of frames near a joint limit (0.5 deg) and beyond one, IK solve
time p50/p95/max (measured with several replays sharing the CPU: comparable
between variants, not a clean benchmark), failed frames and shoulder flips.

* E3: sample-by-sample QP (`sbs`) against the MHE. A bug in its line search
  (the step size was never reset between iterations) was fixed before the
  campaign; it had no measurable effect, because the acceptance test passes at
  the full step. Retuning the QP (up to 10 iterations, gain 1, 5 mm stop) gained
  at most 0.08 deg on participant 1012 at a cost in jitter and solve time, so
  the shipped settings were kept.
* E4: N = 3, 5, 7, 10, 15, 20.
* E5: no filter; 2nd order at 6, 8, 10 Hz; 4th order at 3, 4, 5, 8, 10 Hz. The
  pipeline keeps 4th order at 5 Hz: accuracy does not depend on the setting
  once the delay is compensated, and among settings under 100 ms of delay it
  has the lowest jitter.

## Not produced, by decision of the study lead

Beyond the rough timings above, E6's per-stage and end-to-end latency and the
sustained live run; E7 (single cameras, pairs, triples); E2 (full 43-marker
set); E8 (locomotion tasks); and zero-lag vs lag-compensated accuracy for the
headline table were judged not useful for this paper and were deliberately not
run. (E5 still reports zero-lag accuracy, where the delay is the point.)

## Exclusions and data notes

* Participant 3361 is included in every arm. Its first FastSAM camera-0 export
  (older inference script) had its depth halved and was replaced by a re-export;
  its FastSAM files also carry the videos' swapped camera labels (0<->4, 2<->6),
  the same swap as its mmpose export, measured against mocap and corrected when
  the files are read (`fastsam_source.EXPORT_CAMERA`). Its FastSAM arms were run
  after the main campaign, on the same code (`logs/addendum_3361.log`).
* COMFI's 1118/Screwing camera_0 video was corrupt in an early local copy and has
  been re-downloaded; all runs here use the complete video.

## Figures

{chr(10).join(f"* `figures/{name}.pdf`" for name in figures_made)}

One colour per arm across every figure (`build_handoff.py`, `COLOURS`).
"""


def checklist(paper, figures_made):
    have = lambda p: (paper / p).exists()
    rows = [
        ("E0 filter fix, verified, delay stated", "done",
         "committed (bd09f75); delay in README; section_reconstruction.md corrected"),
        ("E1 modality table, full metric set, per task and DoF group, statistics", "done",
         "tables/joint_overall, joint_tasks, joint_per_dof, markers, freeflyer, stats_joint_rmse; "
         "FastSAM (offline) and an opposed 2-camera pair added"),
        ("Statistics: Friedman, Wilcoxon + Holm, effect size, CI (brief §4)", "done, restricted",
         "Holm over a planned family of 6 comparisons, not all pairs; FastSAM reported outside it"),
        ("Zero-lag scoring as default (brief §4.2.1)", "decided otherwise",
         "lag-compensated scoring kept: COMFI video trails mocap by ~3 frames; residual lag reported"),
        ("COMFI published angles as secondary reference (brief §4)", "decided otherwise",
         "not used at all: different biomechanical model"),
        ("Synchronisation audit (brief §4.2.3)", "done earlier, not in this package",
         "mocap reference vs COMFI published angles: lag 0 on 106/108 trials; video vs mocap offset ~3 frames"),
        ("Zero-lag and lag-compensated accuracy for the headline configuration", "dropped (study lead's decision)",
         "not useful for this paper; E5 reports zero-lag accuracy"),
        ("E3 IK ablation QP vs MHE (accuracy, jitter, limits, solve time, failures)",
         "done" if have("tables/study_ik.md") else "missing", "tables/study_ik, data/studies"),
        ("E4 MHE horizon N = 3..20 (accuracy, jitter, solve time)",
         "done" if have("tables/study_horizon.md") else "missing",
         "tables/study_horizon, figures/study_horizon.pdf; all participants, not a subset"),
        ("E5 filter settings, zero-lag accuracy and delay",
         "done" if have("tables/study_filter.md") else "missing",
         "tables/study_filter, figures/study_filter.pdf; zero-lag and lag-compensated, analytic delay"),
        ("E6 per-stage / end-to-end latency, sustained live run", "dropped (study lead's decision)",
         "rough throughput per arm only (tables/timing) was judged sufficient"),
        ("E7 camera configurations (singles, pairs, triples)", "dropped (study lead's decision)",
         "only the 1/2/4-camera arms"),
        ("E2 full 43-marker set", "dropped (study lead's decision)", "not useful for this paper"),
        ("E8 locomotion tasks", "dropped (study lead's decision)", "not useful for this paper"),
        ("E9 human-robot distance, framed against ISO/TS 15066", "done" if have("tables/robot_body.md") else "missing",
         "tables/robot_*; four distances (the operator hand-guides the robot); separation margin "
         "= p95/p99 overestimate + latency x closing speed"),
        ("Ergonomics: posture REBA agreement (ROADMAP option A)", "done" if have("tables/reba_own.md") else "missing",
         "tables/reba_reference, reba_own"),
        ("REPORT.md", "replaced", "README.md (this package)"),
        ("summary.json", "done" if have("summary.json") else "missing", "summary.json"),
        ("tables/*.tex (booktabs)", "done", "tables/*.tex, generated from the csv"),
        ("per_dof/*.csv", "done" if have("data/per_dof") else "missing", "data/per_dof"),
        ("timing/*.csv|json + environment logs", "done", "data/timing, logs/environment.txt"),
        ("Figure 1: joint RMSE per group and arm, participant dots",
         "done" if "joint_rmse_by_group" in figures_made else "missing", "figures/joint_rmse_by_group.pdf"),
        ("Figure 2: depth vs lateral marker error against camera count",
         "done" if "marker_depth_lateral_vs_cameras" in figures_made else "missing",
         "figures/marker_depth_lateral_vs_cameras.pdf"),
        ("Figure 3: latency breakdown per stage", "replaced",
         "figures/rate_per_arm.pdf (per-stage latency not measured)"),
        ("Figure 4: representative knee and shoulder traces",
         "done" if any(n.startswith("trace_") for n in figures_made) else "missing",
         "figures/trace_*.pdf, data in data/traces"),
    ]
    lines = ["# Checklist against RESULTS_BRIEF.md", "",
             "| Requirement | Status | Where / why |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default="campaign",
                    help="name given to run_campaign.sh (results/<name>, output/<name>)")
    ap.add_argument("--trace-trial", default=None, help="participant/task for the trace figure")
    args = ap.parse_args()

    results = REPO / "results" / args.campaign
    runs_root = REPO / "output" / args.campaign
    paper = results / "paper"
    out = results / "handoff"
    if not (paper / "summary.json").exists():
        raise SystemExit(f"{paper}/summary.json missing: run run_campaign.sh {args.campaign} first")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    summary = json.loads((paper / "summary.json").read_text())
    shutil.copy2(paper / "summary.json", out / "summary.json")

    tables = out / "tables"
    copy_tree(paper / "tables", tables, "*.md")
    copy_tree(paper / "tables", tables, "*.csv")
    for table in sorted((paper / "tables").glob("*.csv")):
        (tables / f"{table.stem}.tex").write_text(
            csv_to_tex(table, f"{table.stem.replace('_', ' ')}: mean (SD) across participants"))

    data = out / "data"
    for sub in ("per_trial", "per_dof", "timing"):
        copy_tree(paper / sub, data / sub, "*.csv")
    copy_tree(paper / "reba", data / "reba", "*.csv")
    copy_tree(paper / "robot_distance", data / "robot_distance", "*.csv")
    (data / "sweep_summaries").mkdir(parents=True, exist_ok=True)
    for source in sorted(list((results / "vs_mocap").glob("*.csv")) + list((results / "timing_runs").glob("*.csv"))):
        rows = read(source)
        name = source.name if source.parent.name == "vs_mocap" else f"timing_run_{source.name}"
        with open(data / "sweep_summaries" / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SWEEP_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    copy_tree(paper / "studies", data / "studies", "*.csv")

    trace_trial = args.trace_trial
    if trace_trial is None:
        participants = sorted({r["participant"] for r in read(paper / "per_trial" / "nlf_0-2-4-6.csv")})
        trace_trial = f"{participants[0]}/Polishing" if participants else "1012/Polishing"
    trace = traces(paper, runs_root, ARMS, trace_trial)
    if trace is not None:
        aligned, _ = trace
        import pandas as pd
        frame = pd.concat({k: v for k, v in aligned.items()}, axis=1)
        frame.columns = [f"{arm}:{dof}" for arm, dof in frame.columns]
        (data / "traces").mkdir(parents=True, exist_ok=True)
        frame.to_csv(data / "traces" / f"{trace_trial.replace('/', '_')}.csv", index_label="frame")
    made = figures(paper, out / "figures", [a for a in ARMS if (paper / "per_trial" / f"{a}.csv").exists()],
                   runs_root, trace_trial)

    copy_tree(results / "logs", out / "logs", "*.log")
    copy_tree(results, out / "logs", "code.diff")
    environment = results / "environment.txt"
    if environment.exists():
        shutil.copy2(environment, out / "logs" / "environment.txt")

    commit = shell("git rev-parse --short HEAD")
    dirty = bool(shell("git status --porcelain"))
    coverage = [(a, summary["coverage"][a]["trials"], summary["coverage"][a]["participants"])
                for a in ARMS if a in summary.get("coverage", {})]
    (out / "README.md").write_text(readme(args.campaign, summary, coverage, made, commit, dirty))
    (out / "CHECKLIST.md").write_text(checklist(out, made))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"handoff written to {out} ({size / 1e6:.1f} MB, {len(made)} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
