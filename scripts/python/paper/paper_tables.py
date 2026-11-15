#!/usr/bin/env python3
"""The tables of the paper, from the run folders and the REBA and hand-distance analyses.

Every figure is the mean (SD) over participants of each participant's mean over
trials. The last table gives, per arm, the commits and landmark-map folds of
the runs behind it.

    python3 scripts/python/paper/paper_tables.py --runs RUNS --reba REBA --hand HAND --out paper-tables.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd

#: Arm folder -> (front end, cameras) as the paper names them.
ARMS = {
    "mmpose_0-2-4-6": ("RTMPose+LSTM", "4"),
    "nlf2d_0-2-4-6": ("NLF-2D", "4"), "nlf2d_0-2": ("NLF-2D", "2S"), "nlf2d_0-4": ("NLF-2D", "2F"),
    "nlf_0-2-4-6": ("NLF-3D", "4"), "nlf_0-2": ("NLF-3D", "2S"), "nlf_0-4": ("NLF-3D", "2F"),
    "nlf_0": ("NLF-3D", "1"),
}
HEADLINE, BASELINE = "nlf_0-2-4-6", "mmpose_0-2-4-6"
GROUPS = {"Lower limbs": ("Hip", "Knee", "Ankle"), "Trunk": ("Lumbar", "Cervical"),
          "Upper limbs": ("Clavicle", "Shoulder", "Elbow")}
PRONATION = "Pronation"


def cell(values, digits=2):
    return f"{values.mean():.{digits}f} ({values.std(ddof=1):.{digits}f})"


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows] + [""]


def load_runs(runs, arm):
    """Per-DoF scores, per-trial scores and provenance of one arm's run folders."""
    dofs, provenance = [], []
    for folder in sorted(runs.glob(f"*/*/{arm}")):
        participant, task = folder.parts[-3], folder.parts[-2]
        frame = pd.read_csv(folder / "per_dof.csv")
        frame["participant"], frame["task"] = participant, task
        dofs.append(frame)
        info = json.loads((folder / "run_info.json").read_text())
        landmark_map = info.get("landmark_map") or {}
        provenance.append(dict(participant=participant, task=task,
                               commit=info["code"]["commit"][:7],
                               dirty=bool(info["code"]["uncommitted"]),
                               fold=landmark_map.get("fold"),
                               in_sample=landmark_map.get("in_sample")))
    return pd.concat(dofs), pd.DataFrame(provenance)


def by_participant(dofs, column, mask=None, absolute=False):
    """Participant means of a per-DoF column averaged over the DoF of each trial."""
    frame = dofs if mask is None else dofs[mask]
    values = frame[column].abs() if absolute else frame[column]
    per_trial = values.groupby([frame.participant, frame.task]).mean()
    return per_trial.groupby("participant").mean()


def accuracy(runs, summaries):
    rows, provenance = [], []
    for arm, (front_end, cameras) in ARMS.items():
        dofs, runs_info = load_runs(runs, arm)
        trials = pd.read_csv(summaries / f"{arm}.csv")
        row = [front_end, cameras, cell(by_participant(dofs, "causal_rmse_deg")),
               cell(by_participant(dofs, "causal_rmse_deg",
                                   ~dofs.dof.str.contains(PRONATION)))]
        for names in GROUPS.values():
            row.append(cell(by_participant(dofs, "causal_rmse_deg",
                                           dofs.dof.str.contains("|".join(names)))))
        row += [cell(by_participant(dofs, "causal_bias_deg", absolute=True)),
                cell(by_participant(dofs, "causal_sd_deg")),
                cell(trials.groupby("participant").causal_marker_mm.mean(), 1),
                f"{trials.groupby('participant').lag.mean().mean():.1f}"]
        rows.append(row)
        folds = runs_info.fold.dropna().astype(int)
        provenance.append([f"{front_end}, {cameras}", len(runs_info),
                           runs_info.participant.nunique(),
                           ", ".join(sorted(runs_info.commit.unique())),
                           int(runs_info.dirty.sum()),
                           f"{folds.nunique()} folds" if len(folds) else "no fitted map",
                           int((runs_info.in_sample == True).sum())])  # noqa: E712
    header = ["Front end", "Cameras", "Whole body", "Without pronation"] + list(GROUPS) + [
        "Mean abs. bias", "SD", "Markers (mm)", "Lag (frames)"]
    return markdown(header, rows), provenance


def per_dof(runs):
    dofs, _ = load_runs(runs, HEADLINE)
    mean = dofs.groupby(["participant", "dof"])[
        ["causal_rmse_deg", "causal_bias_deg", "causal_sd_deg"]].mean().reset_index()
    rows = [[dof, cell(g.causal_rmse_deg), f"{g.causal_bias_deg.mean():+.2f}",
             f"{g.causal_sd_deg.mean():.2f}"] for dof, g in mean.groupby("dof", sort=False)]
    return markdown(["DoF", "RMSE", "Bias", "SD"], rows)


def per_task(runs):
    rows = []
    for arm in (HEADLINE, BASELINE):
        dofs, _ = load_runs(runs, arm)
        trial = dofs.groupby(["participant", "task"]).causal_rmse_deg.mean().reset_index()
        rows.append([ARMS[arm][0]] + [cell(g.causal_rmse_deg) for _, g in trial.groupby("task")])
    tasks = sorted(trial.task.unique())
    return markdown(["Front end"] + tasks, rows)


def reba_tests(root):
    """Wilcoxon signed-rank tests of NLF-3D against RTMPose+LSTM on participant means."""
    from scipy.stats import wilcoxon
    rows = []
    whole, peak = {}, {}
    for arm in (HEADLINE, BASELINE):
        people = pd.read_csv(root / arm / "participants.csv")
        whole[arm] = people[(people.pairing == "causal") & (people.task == "all")
                            & (people.angles == "neutral_own")].set_index("participant")
        chosen = pd.read_csv(root / arm / "representative.csv")
        chosen = chosen[(chosen.selection == "peak") & (chosen.angles == "neutral_own")]
        peak[arm] = (chosen.est_reba - chosen.ref_reba).abs().groupby(chosen.participant).mean()
    for label, a, b in (("MAE over whole trials", whole[HEADLINE].reba_mae, whole[BASELINE].reba_mae),
                        ("Weighted kappa", whole[HEADLINE].risk_kappa, whole[BASELINE].risk_kappa),
                        ("MAE at the peak posture", peak[HEADLINE], peak[BASELINE])):
        a, b = a.align(b, join="inner")
        rows.append([label, f"{a.mean():.2f}", f"{b.mean():.2f}", f"{wilcoxon(a, b).pvalue:.2g}"])
    return markdown(["Quantity (own neutral posture)", "NLF-3D", "RTMPose+LSTM", "p"], rows)


def reba(root):
    rows, peaks = [], []
    for arm, (front_end, cameras) in ARMS.items():
        folder = root / arm
        if not folder.is_dir():
            continue
        people = pd.read_csv(folder / "participants.csv")
        people = people[(people.pairing == "causal") & (people.task == "all")]
        for angles, g in people.groupby("angles", sort=False):
            rows.append([front_end, cameras, angles, cell(g.reba_mae), cell(g.risk_agree_pct, 1),
                         cell(g.risk_kappa), cell(g.high_under_pct.dropna(), 1)])
        chosen = pd.read_csv(folder / "representative.csv")
        chosen = chosen[(chosen.selection == "peak") & (chosen.angles == "neutral_own")]
        chosen = chosen.assign(error=(chosen.est_reba - chosen.ref_reba).abs(),
                               same=100.0 * (chosen.est_risk == chosen.ref_risk))
        for task, g in list(chosen.groupby("task")) + [("All tasks", chosen)]:
            peaks.append([front_end, cameras, task, len(g), cell(g.ref_reba), cell(g.est_reba),
                          cell(g.error), f"{g.same.mean():.1f}"])
    tests = reba_tests(root)
    return (tests, markdown(["Front end", "Cameras", "Angles", "MAE (points)", "Same risk level (%)",
                      "Weighted kappa", "High-risk frames rated lower (%)"], rows),
            markdown(["Front end", "Cameras", "Task", "Postures", "Reference score",
                      "Estimated score", "MAE (points)", "Same risk level (%)"], peaks))


def hand(root):
    pooled = pd.read_csv(root / "pooled.csv")
    pooled = pooled[pooled.task == "all"]
    rows = []
    for arm, (front_end, cameras) in ARMS.items():
        for _, r in pooled[pooled.arm == arm].iterrows():
            rows.append([front_end, cameras, r.hand, r.trials, f"{r.ref_mean_mm:.0f}",
                         f"{r.causal_bias_mm:+.1f}", f"{r.causal_sd_mm:.1f}",
                         f"{r.causal_rmse_mm:.1f}", f"{r.causal_over_p95_mm:.0f}",
                         f"{r.causal_over_p99_mm:.0f}", f"{r.causal_300mm_missed_pct:.2f}",
                         f"{r.causal_300mm_false_alarm_pct:.2f}",
                         f"{r.causal_105mm_missed_pct:.0f}"])
    return markdown(["Front end", "Cameras", "Hand", "Trials", "Reference distance (mm)",
                     "Bias (mm)", "SD (mm)", "RMSE (mm)", "Overstated by, 95th pct. (mm)",
                     "99th pct. (mm)", "Missed at 300 mm (%)", "False alarms at 300 mm (%)",
                     "Missed at 105 mm (%)"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="<participant>/<task>/<arm> folders")
    parser.add_argument("--summaries", type=Path, required=True, help="one <arm>.csv per arm")
    parser.add_argument("--reba", type=Path, required=True, help="output of reba_analyses.py")
    parser.add_argument("--hand", type=Path, required=True, help="output of hand_distance.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table, provenance = accuracy(args.runs, args.summaries)
    tests, agreement, peaks = reba(args.reba)
    lines = ["# Tables of the paper (COMFI)", "",
             "Mean (SD) over the 18 participants; errors are causal (no time alignment). "
             "Produced by `scripts/python/paper/paper_tables.py`.", "",
             "## Joint-angle error (deg) and marker error", ""] + table
    lines += ["## Per degree of freedom, NLF-3D, four cameras (deg)", ""] + per_dof(args.runs)
    lines += ["## Per task, four cameras (deg)", ""] + per_task(args.runs)
    lines += ["## Posture-only REBA over whole trials", ""] + agreement
    lines += ["## Posture-only REBA, NLF-3D against RTMPose+LSTM, four cameras (not corrected)",
              ""] + tests
    lines += ["## Posture-only REBA at the peak posture picked on the estimate "
              "(own neutral posture)", ""] + peaks
    lines += ["## Hand to end effector on the robot tasks, all trials pooled", ""] + hand(args.hand)
    lines += ["## Runs behind each arm", ""] + markdown(
        ["Arm", "Trials", "Participants", "Commit", "Runs with uncommitted code",
         "Landmark maps", "Runs with an in-sample map"], provenance)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
