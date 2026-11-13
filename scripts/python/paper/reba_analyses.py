#!/usr/bin/env python3
"""Posture-only REBA analyses of arms against the reference, over every trial found.

For each arm, under ``<out>/<arm>/``:

  trials.csv          agreement per trial, pairing and angle reference
  participants.csv    the same with each participant's frames pooled
  summary.csv         participant mean, SD and n of every figure, and all frames pooled
  confusion.csv       risk-level counts and shares per reference level
  representative.csv  postures picked on the estimate (causal pairing), reference beside
  frames/             per-frame REBA of estimate and reference (causal)
  summary.md          the tables above, condensed

    python3 scripts/python/paper/reba_analyses.py --runs /root/workspace/oof/runs \\
        --dataset /root/workspace/COMFI --out /root/workspace/oof/t12 \\
        --arms nlf_0-2-4-6 fusion_mean
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

KEYS = ["arm", "pairing", "angles"]


def trials_of(runs, arm):
    return sorted((d.parts[-3], d.parts[-2], d) for d in Path(runs).glob(f"*/*/{arm}")
                  if (d / "joint_angles.csv").exists())


def out_of_fold(run_dir):
    """Whether the run's landmark map was fitted without its participant; None if unrecorded."""
    path = run_dir / "run_info.json"
    info = json.loads(path.read_text()).get("landmark_map") if path.exists() else None
    return None if not isinstance(info, dict) or "in_sample" not in info else not info["in_sample"]


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def peak_summary(ra, rows):
    """How the reference rates the estimate's peak postures: participant mean, SD, n."""
    by = {}
    for r in rows:
        if r["selection"] == "peak" and np.isfinite(r["ref_reba"]):
            by.setdefault(r["participant"], []).append(r)
    stat = lambda f: ra.mean_sd([np.mean([f(r) for r in v]) for v in by.values()])
    return {"trials": sum(len(v) for v in by.values()),
            "est_reba": stat(lambda r: r["est_reba"]), "ref_reba": stat(lambda r: r["ref_reba"]),
            "error": stat(lambda r: r["est_reba"] - r["ref_reba"]),
            "same_level_pct": stat(lambda r: 100.0 * (r["est_risk"] == r["ref_risk"])),
            "ref_higher_pct": stat(lambda r: 100.0 * (r["ref_risk"] > r["est_risk"])),
            "ref_lower_pct": stat(lambda r: 100.0 * (r["ref_risk"] < r["est_risk"]))}


def analyse(ra, settings, dataset, arm, trials, out):
    from rtcosmik.ergonomics.reba_posture import RISK_NAMES
    from rtcosmik.paper import scoring
    (out / "frames").mkdir(parents=True, exist_ok=True)
    hold = round(settings.paper["reba_hold_s"] * settings.fs)
    combos = [(p, a) for p in ra.PAIRINGS for a in ra.ANGLE_REFERENCES]
    series = {c: [] for c in combos}
    trial_rows, postures, lags, provenance, failed = [], [], [], set(), []
    for participant, task, run_dir in trials:
        try:
            pairs, lag = ra.paired_scores(run_dir, scoring.reference_dir(dataset, participant, task),
                                          scoring.clock_offset(settings, participant, task))
        except Exception as exc:
            failed.append(f"{participant}/{task}: {type(exc).__name__}: {exc}")
            continue
        provenance.add(out_of_fold(run_dir))
        lags.append(lag)
        frames = {}
        for (pairing, angles), pair in pairs.items():
            est, ref = pair["est"]["reba"], pair["ref"]["reba"]
            series[pairing, angles].append((participant, task, est, ref))
            trial_rows.append({"arm": arm, "participant": participant, "task": task, "lag": lag,
                               "pairing": pairing, "angles": angles, **ra.agreement(est, ref)})
            if pairing == "causal":
                frames.update({"frame": pair["frame"], f"{angles}_est": est, f"{angles}_ref": ref})
                postures += [{"arm": arm, "participant": participant, "task": task,
                              "angles": angles, **row}
                             for row in ra.representative_postures(pair, hold)]
        np.savez_compressed(out / "frames" / f"{participant}_{task}.npz", **frames)

    participant_rows, summary_rows, confusion_rows, tables = [], [], [], {}
    for (pairing, angles), items in series.items():
        key = {"arm": arm, "pairing": pairing, "angles": angles}
        for task in ["all"] + sorted({t for _, t, _, _ in items}):
            chosen = [(p, e, r) for p, t, e, r in items if task in ("all", t)]
            participants, across, pooled = ra.summarize(chosen)
            participant_rows += [{**key, "task": task, "participant": p, **a}
                                 for p, a in participants.items()]
            row = {**key, "task": task, "participants": len(participants)}
            for metric, (mean, sd, n) in across.items():
                row.update({f"{metric}_mean": mean, f"{metric}_sd": sd, f"{metric}_n": n,
                            f"{metric}_pooled": pooled[metric]})
            summary_rows.append(row)
            if task == "all":
                tables[pairing, angles] = (across, pooled)
                shares = ra.by_reference_level(pooled["confusion"])
                confusion_rows += [{**key, "reference": RISK_NAMES[i], "estimate": RISK_NAMES[j],
                                    "frames": int(pooled["confusion"][i, j]),
                                    "share_of_reference_pct": 100 * shares[i, j]}
                                   for i in range(ra.LEVELS) for j in range(ra.LEVELS)]

    write(out / "trials.csv", KEYS + ["participant", "task", "lag", *ra.SCALARS], trial_rows)
    write(out / "participants.csv", KEYS + ["task", "participant", *ra.SCALARS], participant_rows)
    write(out / "summary.csv", list(summary_rows[0]), summary_rows)
    write(out / "confusion.csv", KEYS + ["reference", "estimate", "frames",
                                         "share_of_reference_pct"], confusion_rows)
    write(out / "representative.csv", list(postures[0]), postures)
    peaks = {a: peak_summary(ra, [r for r in postures if r["angles"] == a])
             for a in ra.ANGLE_REFERENCES}
    report(out / "summary.md", arm, ra, RISK_NAMES, tables, peaks, hold, len(lags), lags,
           provenance, failed)
    return len(lags), failed


def report(path, arm, ra, levels, tables, peaks, hold, trials, lags, provenance, failed):
    pm = lambda t, d=2: f"{t[0]:.{d}f} ± {t[1]:.{d}f}"
    final = provenance == {True}
    lines = [f"# Posture-only REBA: {arm}", "",
             f"{trials} trials; lag beyond the clock offset {np.mean(lags):.1f} ± "
             f"{np.std(lags):.1f} frames. Landmark maps: "
             + ("out-of-fold." if final else "NOT all recorded as out-of-fold: not final."),
             "Figures are participant mean ± SD, each participant's frames pooled.", ""]
    if failed:
        lines += ["Failed trials:", *[f"- {f}" for f in failed], ""]
    lines += ["## Agreement", "",
              "| pairing | angles | REBA est | REBA ref | bias | MAE | exact % | risk agree % "
              "| weighted kappa |", "|---|---|---|---|---|---|---|---|---|"]
    for (pairing, angles), (a, _) in tables.items():
        lines.append(f"| {pairing} | {angles} | {pm(a['reba_est_mean'])} | "
                     f"{pm(a['reba_ref_mean'])} | {pm(a['reba_bias'])} | {pm(a['reba_mae'])} | "
                     f"{pm(a['reba_exact_pct'], 1)} | {pm(a['risk_agree_pct'], 1)} | "
                     f"{pm(a['risk_kappa'])} |")
    lines += ["", "## Frames the reference rates high or very high", "",
              "| pairing | angles | participants | frames | rated lower % | rated below high % "
              "| signed error | pooled rated lower % |", "|---|---|---|---|---|---|---|---|"]
    for (pairing, angles), (a, pooled) in tables.items():
        lines.append(f"| {pairing} | {angles} | {a['high_under_pct'][2]} | "
                     f"{pooled['high_frames']} | {pm(a['high_under_pct'], 1)} | "
                     f"{pm(a['high_missed_pct'], 1)} | {pm(a['high_bias'])} | "
                     f"{pooled['high_under_pct']:.1f} |")
    for (pairing, angles), (_, pooled) in tables.items():
        if pairing != "causal":
            continue
        shares = ra.by_reference_level(pooled["confusion"])
        lines += ["", f"## Confusion, causal, {angles}", "",
                  "Rows: reference level. Columns: estimate level. Frames (share of the row).", "",
                  "| reference | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
        for i, name in enumerate(levels):
            cells = [f"{pooled['confusion'][i, j]} ({100 * shares[i, j]:.1f}%)"
                     if pooled["confusion"][i].sum() else "0" for j in range(len(levels))]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Peak postures picked on the estimate", "",
              f"One per trial: the middle of the longest spell at the highest REBA the estimate "
              f"holds for {hold} frames; causal pairing.", "",
              "| angles | trials | REBA est | REBA ref | est - ref | same level % "
              "| reference higher % | reference lower % |", "|---|---|---|---|---|---|---|---|"]
    for angles, p in peaks.items():
        lines.append(f"| {angles} | {p['trials']} | {pm(p['est_reba'])} | {pm(p['ref_reba'])} | "
                     f"{pm(p['error'])} | {pm(p['same_level_pct'], 1)} | "
                     f"{pm(p['ref_higher_pct'], 1)} | {pm(p['ref_lower_pct'], 1)} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True,
                    help="root holding <participant>/<task>/<arm>/joint_angles.csv")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset root holding the reference runs")
    ap.add_argument("--arms", nargs="+", required=True, help="arm folder names under --runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    from rtcosmik.paper import reba_analysis as ra
    status = 0
    for arm in args.arms:
        trials = trials_of(args.runs, arm)
        if not trials:
            print(f"{arm}: no trials under {args.runs}")
            status = 1
            continue
        done, failed = analyse(ra, settings, args.dataset, arm, trials, args.out / arm)
        print(f"{arm}: {done} trials -> {args.out / arm}"
              + (f", {len(failed)} failed" if failed else ""), flush=True)
        for line in failed:
            print("  " + line)
    return status


if __name__ == "__main__":
    sys.exit(main())
