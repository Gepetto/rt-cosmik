#!/usr/bin/env python3
"""Turn the per-trial tables into the paper's numbers and statistics.

Reads ``per_trial/<arm>.csv`` and ``per_dof/<arm>.csv`` written by
``trial_metrics.py`` and produces, under ``tables/``:

* ``joint_overall``   RMSE, |bias|, SD of the error and r, whole body and per DoF group
* ``joint_tasks``     whole-body RMSE per task
* ``joint_per_dof``   the same four numbers for every scored DoF (too long for the
                      paper; kept for the record)
* ``markers``         marker error RMS, split depth / lateral and translation / shape
* ``freeflyer``       root position and orientation error, and residual lag
* ``reba_<neutral>``  posture-REBA agreement with the reference (``reba_agreement.py``),
                      including a linear-weighted Cohen's kappa on risk levels, pooled
                      over each participant's frames
* ``robot_<measure>`` human-robot distance error (``robot_distance.py``), including the
                      separation margin a monitor would need: the 95th / 99th percentile
                      of the overestimated distance, plus latency x closing speed
* ``timing``          throughput and IK time from the sweep summaries (for replayed
                      arms, from their direct timing runs); FastSAM's inference time
                      per view from its logs (``fastsam_timing.py``)
* ``study_ik``, ``study_horizon``, ``study_filter``
                      E3 / E4 / E5 on the NLF-3D 4-camera pipeline
                      (``ik_filter_studies.py``), each variant against the baseline
* ``stats_*``         Friedman across the real-time arms, then the planned comparisons
                      (:data:`PLANNED`): Wilcoxon signed-rank with Holm correction,
                      rank-biserial effect size, bootstrap 95% CI of the paired
                      difference. FastSAM is offline and outside that family: its
                      comparisons at matched camera counts are reported uncorrected.

and ``summary.json`` holding all of it.

How numbers are aggregated, everywhere:

1. Each trial gives one value per DoF.
2. A participant's value is the mean over their trials (and over the DoF in the
   group, where a group is reported).
3. The table reports mean (SD) across participants, SD with ddof = 1.

Bias is reported as the mean *absolute* per-DoF bias. Signed biases of different
DoF cancel when averaged, which would make a biased arm look unbiased.

Statistics use one value per participant, never per trial: trials from the same
participant are not independent, and treating 108 trials as 108 observations
overstates significance. Each comparison uses the participants both arms cover,
and the count is stated; the Friedman test uses those all its arms cover.

    python3 scripts/python/paper/aggregate_results.py --arms nlf_0 fastsam_0 ...
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

import numpy as np

GROUPS = ("lower", "trunk", "upper")
TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
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
#: The planned comparisons, Holm-corrected as one family: the proposed pipeline
#: against the previous architecture and against triangulating the same
#: detector's 2D, the camera-count ladder, and camera placement.
PLANNED = [("nlf_0-2-4-6", "mmpose_0-2-4-6"), ("nlf_0-2-4-6", "nlf2d_0-2-4-6"),
           ("nlf_0", "nlf_0-2"), ("nlf_0-2", "nlf_0-2-4-6"), ("nlf_0", "nlf_0-2-4-6"),
           ("nlf_0-2", "nlf_0-4")]
#: FastSAM (offline) against NLF-3D at the same cameras; reported, not corrected.
OFFLINE = [("fastsam_0", "nlf_0"), ("fastsam_0-2", "nlf_0-2"), ("fastsam_0-4", "nlf_0-4"),
           ("fastsam_0-2-4-6", "nlf_0-2-4-6")]
MARKER_FIELDS = ("marker_raw_mm", "marker_depth_mm", "marker_lateral_mm",
                 "marker_translation_mm", "marker_shape_mm")
#: Participants excluded from marker geometry for every arm (FastSAM export defect).
MARKER_EXCLUDED = {"3361"}
FPS = 40.0
#: RTMPose 2D inference, mean (SD) ms per multi-view frame, from Table I of the
#: RT-COSMIK draft (RTX 4500 Ada, i9-14900K). The mmpose arm reads COMFI's
#: precomputed 2D keypoints, so its sweep never pays this cost.
RTMPOSE_MS = {2: (7.1, 0.9), 4: (13.2, 0.7)}


def read(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def mean_sd(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def participant_means(rows, value, where=lambda r: True):
    """Participant -> mean of ``value(row)`` over the rows that pass ``where``."""
    buckets = defaultdict(list)
    for r in rows:
        if where(r):
            x = value(r)
            if np.isfinite(x):
                buckets[r["participant"]].append(x)
    return {p: float(np.mean(v)) for p, v in buckets.items()}


def dof_participant_means(dof_rows, field, group=None, absolute=False):
    """Mean over DoF within a trial, then over trials within a participant."""
    per_trial = defaultdict(list)
    for r in dof_rows:
        if group and r["group"] != group:
            continue
        x = float(r[field])
        if np.isfinite(x):
            per_trial[(r["participant"], r["task"])].append(abs(x) if absolute else x)
    per_participant = defaultdict(list)
    for (p, _), values in per_trial.items():
        per_participant[p].append(float(np.mean(values)))
    return {p: float(np.mean(v)) for p, v in per_participant.items()}


def weighted_kappa(a, b, levels):
    """Linear-weighted Cohen's kappa between two integer label sequences."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.size == 0:
        return np.nan
    observed = np.zeros((levels, levels))
    np.add.at(observed, (a, b), 1)
    observed /= observed.sum()
    expected = np.outer(observed.sum(1), observed.sum(0))
    i, j = np.indices((levels, levels))
    weights = np.abs(i - j) / (levels - 1)
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator > 0 else np.nan


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def friedman(by_arm, arms):
    """Friedman test over ``arms``, on the participants every one of them covers."""
    from scipy import stats
    arms = [a for a in arms if by_arm.get(a)]
    common = sorted(set.intersection(*(set(by_arm[a]) for a in arms))) if arms else []
    if len(arms) < 3 or len(common) < 5:
        return {"arms": arms, "n": len(common), "chi2": None, "p": None, "df": None}
    chi2, p = stats.friedmanchisquare(*np.array([[by_arm[a][q] for a in arms] for q in common]).T)
    return {"arms": arms, "n": len(common), "chi2": float(chi2), "p": float(p), "df": len(arms) - 1}


def compare(by_arm, pairs, rng, correct):
    """Wilcoxon signed-rank for each pair, on the participants both arms cover.

    Also the mean paired difference with a bootstrap 95% CI and the rank-biserial
    effect size. With ``correct``, p-values are Holm-adjusted across ``pairs``.
    """
    from scipy import stats
    out = []
    for a, b in pairs:
        if not (by_arm.get(a) and by_arm.get(b)):
            continue
        common = sorted(set(by_arm[a]) & set(by_arm[b]))
        if len(common) < 5:
            continue
        diff = np.array([by_arm[a][q] - by_arm[b][q] for q in common])
        p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
        nonzero = diff[diff != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        rbc = (float(ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / float(ranks.sum())
               if ranks.size else 0.0)
        boots = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        out.append({"a": a, "b": b, "n": len(common), "mean_diff": float(diff.mean()),
                    "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "p": p, "rank_biserial": rbc})
    adjusted = holm(np.asarray([c["p"] for c in out])) if (correct and out) else [c["p"] for c in out]
    for c, p_adj in zip(out, adjusted):
        c["p_adjusted"] = float(p_adj)
        c["correction"] = "Holm" if correct else "none"
        c["significant_0.05"] = bool(p_adj < 0.05)
    return out


def filter_delay_ms(frequency_hz=1.0):
    """Group delay of the configured marker filter at ``frequency_hz``, in ms."""
    sys.path.insert(0, str(REPO / "src"))
    from scipy import signal
    from rtcosmik.config_loader import settings
    b, a = signal.butter(settings.order, settings.cutoff_freq, settings.filter_type, fs=settings.fs)
    _, delay = signal.group_delay((b, a), w=[frequency_hz], fs=settings.fs)
    return float(delay[0] / settings.fs * 1000.0)


def fmt(stat, digits=2):
    if stat["mean"] is None:
        return "--"
    return f"{stat['mean']:.{digits}f} ({stat['sd']:.{digits}f})"


def write_table(path_stem, header, rows):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    with open(path_stem.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    path_stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=REPO / "results" / "paper")
    ap.add_argument("--results", type=Path, default=None,
                    help="folder whose vs_mocap/ holds the sweep summaries; "
                         "defaults to the parent of --root")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    trials = {a: read(args.root / "per_trial" / f"{a}.csv") for a in args.arms}
    dofs = {a: read(args.root / "per_dof" / f"{a}.csv") for a in args.arms}
    arms = [a for a in args.arms if trials[a]]
    missing = [a for a in args.arms if not trials[a]]
    tables = args.root / "tables"
    summary = {"arms": {a: LABELS.get(a, a) for a in arms}, "missing_arms": missing,
               "aggregation": "per-DoF value per trial -> participant mean -> mean (SD, ddof=1) across participants",
               "coverage": {}, "joint": {}, "tasks": {}, "markers": {}, "freeflyer": {},
               "lag": {}, "per_dof": {}, "reba": {}, "robot_distance": {}, "timing": {},
               "stats": {}}
    results = args.results or args.root.parent

    for a in arms:
        summary["coverage"][a] = {"trials": len(trials[a]),
                                  "participants": len({r["participant"] for r in trials[a]})}

    # Joint angles: whole body and per group.
    joint_rows, by_arm_rmse = [], {}
    for a in arms:
        entry = {}
        for scope in (None,) + GROUPS:
            key = scope or "all"
            rmse = dof_participant_means(dofs[a], "rmse_deg", scope)
            entry[key] = {
                "rmse_deg": mean_sd(rmse.values()),
                "abs_bias_deg": mean_sd(dof_participant_means(dofs[a], "bias_deg", scope, True).values()),
                "sd_deg": mean_sd(dof_participant_means(dofs[a], "sd_deg", scope).values()),
                "r": mean_sd(dof_participant_means(dofs[a], "r", scope).values()),
            }
            if scope is None:
                by_arm_rmse[a] = rmse
        summary["joint"][a] = entry
        row = [LABELS.get(a, a)]
        for key in ("all",) + GROUPS:
            e = entry[key]
            row += [fmt(e["rmse_deg"]), fmt(e["abs_bias_deg"]), fmt(e["sd_deg"]), fmt(e["r"], 3)]
        joint_rows.append(row)
    header = ["arm"] + [f"{k} {m}" for k in ("all",) + GROUPS
                        for m in ("RMSE deg", "|bias| deg", "SD deg", "r")]
    write_table(tables / "joint_overall", header, joint_rows)

    # Per task.
    task_rows = []
    for a in arms:
        per = {t: mean_sd(participant_means(trials[a], lambda r: float(r["joint_rmse_deg"]),
                                            lambda r, t=t: r["task"] == t).values())
               for t in TASKS}
        summary["tasks"][a] = per
        task_rows.append([LABELS.get(a, a)] + [fmt(per[t]) for t in TASKS])
    write_table(tables / "joint_tasks", ["arm"] + list(TASKS), task_rows)

    # Per DoF, for the record.
    dof_rows = []
    for a in arms:
        names = sorted({r["dof"] for r in dofs[a]})
        summary["per_dof"][a] = {}
        for name in names:
            sel = [r for r in dofs[a] if r["dof"] == name]
            stat = {f: mean_sd(participant_means(sel, lambda r, f=f: abs(float(r[f])) if f == "bias_deg"
                                                 else float(r[f])).values())
                    for f in ("rmse_deg", "bias_deg", "sd_deg", "r")}
            summary["per_dof"][a][name] = stat
            dof_rows.append([LABELS.get(a, a), name, fmt(stat["rmse_deg"]), fmt(stat["bias_deg"]),
                             fmt(stat["sd_deg"]), fmt(stat["r"], 3)])
    write_table(tables / "joint_per_dof", ["arm", "dof", "RMSE deg", "|bias| deg", "SD deg", "r"],
                dof_rows)

    # Markers, root pose, lag.
    marker_rows, root_rows = [], []
    for a in arms:
        keep = lambda r: r["participant"] not in MARKER_EXCLUDED
        m = {f: mean_sd(participant_means(trials[a], lambda r, f=f: float(r[f]), keep).values())
             for f in MARKER_FIELDS}
        depth, lateral = m["marker_depth_mm"]["mean"], m["marker_lateral_mm"]["mean"]
        m["anisotropy"] = depth / lateral if depth and lateral else None
        summary["markers"][a] = m
        marker_rows.append([LABELS.get(a, a)] + [fmt(m[f], 1) for f in MARKER_FIELDS]
                           + [f"{m['anisotropy']:.2f}" if m["anisotropy"] else "--"])
        ff = {f: mean_sd(participant_means(trials[a], lambda r, f=f: float(r[f])).values())
              for f in ("freeflyer_pos_mm", "freeflyer_rot_deg")}
        lag = mean_sd(participant_means(trials[a], lambda r: float(r["lag_frames"])).values())
        summary["freeflyer"][a], summary["lag"][a] = ff, {
            "frames": lag, "ms_mean": lag["mean"] * 1000 / FPS if lag["mean"] is not None else None}
        flips = {t: mean_sd(participant_means(trials[a], lambda r: float(r.get("shoulder_flip_pct") or "nan"),
                                              lambda r, t=t: t is None or r["task"] == t).values())
                 for t in (None, "SideOverhead")}
        summary["freeflyer"][a]["shoulder_flip_pct"] = flips[None]
        summary["freeflyer"][a]["shoulder_flip_pct_side_overhead"] = flips["SideOverhead"]
        root_rows.append([LABELS.get(a, a), fmt(ff["freeflyer_pos_mm"], 1),
                          fmt(ff["freeflyer_rot_deg"], 2), fmt(lag, 1),
                          fmt(flips[None], 1), fmt(flips["SideOverhead"], 1)])
    write_table(tables / "markers", ["arm", "raw mm", "depth mm", "lateral mm",
                                     "translation mm", "shape mm", "depth/lateral"], marker_rows)
    write_table(tables / "freeflyer", ["arm", "root position mm", "root orientation deg",
                                       "residual lag frames", "shoulder flip % (all tasks)",
                                       "shoulder flip % (SideOverhead)"], root_rows)

    # Posture REBA, per neutral mode.
    reba_fields = ("reba_mae", "reba_bias", "reba_exact_pct", "risk_agree_pct",
                   "time_in_level_err_pct", "neck_agree_pct", "trunk_agree_pct",
                   "legs_agree_pct", "upper_arm_agree_pct", "lower_arm_agree_pct")
    by_arm_reba = {}
    for mode in ("reference", "own"):
        rows_out = []
        for a in arms:
            sel = [r for r in read(args.root / "reba" / f"{a}.csv") if r["neutral"] == mode]
            if not sel:
                continue
            entry = {f: mean_sd(participant_means(sel, lambda r, f=f: float(r[f])).values())
                     for f in reba_fields}
            pooled = defaultdict(list)
            for r in sel:
                path = args.root / "reba" / "levels" / a / f"{r['participant']}_{r['task']}_{mode}.npy"
                if path.exists():
                    pooled[r["participant"]].append(np.load(path))
            kappas = {p: weighted_kappa(np.concatenate(v)[:, 0], np.concatenate(v)[:, 1], 5)
                      for p, v in pooled.items()}
            entry["risk_kappa_linear"] = mean_sd(kappas.values())
            entry["reba_ref_mean"] = mean_sd(participant_means(sel, lambda r: float(r["reba_ref_mean"])).values())
            summary["reba"].setdefault(mode, {})[a] = entry
            if mode == "own":
                by_arm_reba[a] = participant_means(sel, lambda r: float(r["reba_mae"]))
            rows_out.append([LABELS.get(a, a), fmt(entry["reba_mae"]), fmt(entry["reba_bias"]),
                             fmt(entry["reba_exact_pct"], 1), fmt(entry["risk_agree_pct"], 1),
                             fmt(entry["risk_kappa_linear"], 2), fmt(entry["time_in_level_err_pct"], 1)]
                            + [fmt(entry[f"{c}_agree_pct"], 1)
                               for c in ("neck", "trunk", "legs", "upper_arm", "lower_arm")])
        if rows_out:
            write_table(tables / f"reba_{mode}",
                        ["arm", "REBA MAE", "REBA bias", "exact %", "risk agree %", "risk kappa",
                         "time-in-level err %", "neck %", "trunk %", "legs %", "upper arm %",
                         "lower arm %"], rows_out)

    # Timing. Offline throughput of each arm's sweep, IK time, and FastSAM's
    # inference per view, which the sweep does not see (it reads exported markers).
    fastsam_views = read(args.root / "timing" / "fastsam_per_view.csv")
    view_ms = mean_sd(participant_means(fastsam_views, lambda r: float(r["mean_ms"])).values())
    summary["timing"]["fastsam_inference_per_view_ms"] = view_ms
    timing_rows = []
    for a in arms:
        sweep_rows = [r for r in read(results / "vs_mocap" / f"{a}.csv") if r.get("fps")]
        if not sweep_rows:          # replayed arm: its throughput comes from direct timing runs
            sweep_rows = [r for r in read(results / "timing_runs" / f"{a}.csv")
                          if r.get("status") == "ok" and r.get("fps")]
        fps = mean_sd(participant_means(sweep_rows, lambda r: float(r["fps"])).values())
        ik = mean_sd(participant_means(sweep_rows, lambda r: float(r["ik_ms_median"] or "nan")).values())
        ik95 = mean_sd(participant_means(sweep_rows, lambda r: float(r["ik_ms_p95"] or "nan")).values())
        entry = {"sweep_fps": fps, "ik_ms_median": ik, "ik_ms_p95": ik95}
        if a.startswith("fastsam") and view_ms["mean"] is not None:
            cams = len(a.split("_", 1)[1].split("-"))
            per_frame = cams * view_ms["mean"] + (ik["mean"] or 0.0)
            entry["note"] = f"{cams} x {view_ms['mean']:.0f} ms inference per view (sequential) + IK"
            entry["estimated_rate_hz"] = 1000.0 / per_frame
        elif a.startswith("mmpose") and fps["mean"]:
            cams = len(a.split("_", 1)[1].split("-"))
            hpe = RTMPOSE_MS.get(cams, (np.nan, np.nan))[0]
            entry["note"] = (f"{hpe} ms RTMPose 2D inference (draft Table I) + measured "
                             "triangulation, LSTM, filter, IK")
            entry["estimated_rate_hz"] = 1000.0 / (hpe + 1000.0 / fps["mean"])
        else:
            entry["note"] = "end to end from video files: decode, detection, NLF, fusion, filter, IK"
            entry["estimated_rate_hz"] = fps["mean"]
        rate = entry["estimated_rate_hz"]
        entry["realtime_30hz"] = bool(rate is not None and rate >= 30.0)
        summary["timing"][a] = entry
        timing_rows.append([LABELS.get(a, a), fmt(fps, 1), fmt(ik, 2), fmt(ik95, 2),
                            f"{rate:.1f}" if rate is not None else "--",
                            "yes" if entry["realtime_30hz"] else "no", entry["note"]])
    write_table(tables / "timing", ["arm", "sweep fps", "IK ms (median)", "IK ms (p95)",
                                    "estimated rate Hz", ">= 30 Hz", "what it covers"], timing_rows)

    # Human-robot distance, per measure.
    # The margin a speed-and-separation monitor would add on top of the estimate:
    # the 95th/99th percentile of overestimated distance, plus the distance the
    # operator closes during the pipeline's latency (filter group delay at 1 Hz
    # plus one frame of processing at the arm's rate; camera exposure and
    # transport not included) at the reference's 95th-percentile closing speed.
    robot_fields = ("bias_mm", "mae_mm", "rmse_mm", "sd_mm", "r", "closest_approach_err_mm",
                    "unsafe_p95_mm", "unsafe_p99_mm", "closing_speed_p95_mm_s",
                    "contact_agree_pct", "below_100mm_agree_pct", "below_200mm_agree_pct",
                    "below_300mm_agree_pct", "closest_segment_agree_pct", "ref_mean_mm", "ref_min_mm")
    delay_ms = filter_delay_ms()
    summary["filter_delay_1hz_ms"] = delay_ms
    by_arm_body, by_arm_hands = {}, {}
    for measure in ("whole", "body", "left_hand_ee", "right_hand_ee"):
        rows_out = []
        for a in arms:
            sel = [r for r in read(args.root / "robot_distance" / f"{a}.csv") if r["measure"] == measure]
            if not sel:
                continue
            entry = {f: mean_sd(participant_means(sel, lambda r, f=f: float(r.get(f) or "nan")).values())
                     for f in robot_fields}
            rate = summary["timing"].get(a, {}).get("estimated_rate_hz")
            latency_ms = delay_ms + (1000.0 / rate if rate else np.nan)
            speed = entry["closing_speed_p95_mm_s"]["mean"]
            entry["latency_ms"] = latency_ms
            entry["latency_margin_mm"] = latency_ms / 1000.0 * speed if speed is not None else None
            p95 = entry["unsafe_p95_mm"]["mean"]
            entry["total_margin_p95_mm"] = (max(p95, 0.0) + entry["latency_margin_mm"]
                                            if p95 is not None and entry["latency_margin_mm"] is not None
                                            else None)
            summary["robot_distance"].setdefault(measure, {})[a] = entry
            if measure == "body":
                by_arm_body[a] = participant_means(sel, lambda r: float(r["rmse_mm"]))
            if measure.endswith("hand_ee"):
                for p, v in participant_means(sel, lambda r: float(r["rmse_mm"])).items():
                    by_arm_hands.setdefault(a, {}).setdefault(p, []).append(v)
            number = lambda x, d=0: f"{x:.{d}f}" if x is not None and np.isfinite(x) else "--"
            rows_out.append([LABELS.get(a, a), fmt(entry["bias_mm"], 1), fmt(entry["mae_mm"], 1),
                             fmt(entry["rmse_mm"], 1), fmt(entry["sd_mm"], 1), fmt(entry["r"], 3),
                             fmt(entry["unsafe_p95_mm"], 1), fmt(entry["unsafe_p99_mm"], 1),
                             number(entry["latency_ms"]), number(entry["latency_margin_mm"], 1),
                             number(entry["total_margin_p95_mm"], 1),
                             fmt(entry["closest_approach_err_mm"], 1), fmt(entry["contact_agree_pct"], 1),
                             fmt(entry["below_100mm_agree_pct"], 1), fmt(entry["below_200mm_agree_pct"], 1),
                             fmt(entry["below_300mm_agree_pct"], 1),
                             fmt(entry["closest_segment_agree_pct"], 1)])
        if rows_out:
            write_table(tables / f"robot_{measure}",
                        ["arm", "bias mm", "MAE mm", "RMSE mm", "SD mm", "r",
                         "overestimate p95 mm", "overestimate p99 mm", "latency ms",
                         "latency margin mm", "total margin (p95) mm", "closest approach err mm",
                         "contact agree %", "<100 mm agree %", "<200 mm agree %", "<300 mm agree %",
                         "closest segment agree %"], rows_out)
    by_arm_hands = {a: {p: float(np.mean(v)) for p, v in d.items() if len(v) == 2}
                    for a, d in by_arm_hands.items()}

    # Statistics, one value per participant.
    realtime = [a for a in arms if not a.startswith("fastsam")]

    def stats_table(name, by_arm, unit, digits=2):
        stat = {"friedman": friedman(by_arm, realtime),
                "planned": compare(by_arm, PLANNED, rng, correct=True),
                "offline": compare(by_arm, OFFLINE, rng, correct=False)}
        summary["stats"][name] = stat
        rows = [["family", "a", "b", "n", f"mean diff (a - b) {unit}", "95% CI", "p", "p adjusted",
                 "correction", "rank-biserial", "sig"]]
        for family in ("planned", "offline"):
            for c in stat[family]:
                rows.append([family, LABELS.get(c["a"], c["a"]), LABELS.get(c["b"], c["b"]), c["n"],
                             f"{c['mean_diff']:+.{digits}f}",
                             f"[{c['ci95'][0]:+.{digits}f}, {c['ci95'][1]:+.{digits}f}]",
                             f"{c['p']:.4f}", f"{c['p_adjusted']:.4f}", c["correction"],
                             f"{c['rank_biserial']:+.2f}", "yes" if c["significant_0.05"] else "no"])
        f = stat["friedman"]
        rows.append(["friedman", f"{len(f['arms'])} real-time arms", "", f["n"], "", "",
                     f"{f['p']:.2e}" if f["p"] is not None else "--", "", "",
                     f"chi2({f['df']}) = {f['chi2']:.1f}" if f["chi2"] is not None else "--", ""])
        write_table(tables / f"stats_{name}", rows[0], rows[1:])
        return stat

    # Design-choice studies (E3 IK type, E4 horizon, E5 filter).
    study_fields = ("joint_rmse_deg", "joint_rmse_zero_lag_deg", "upper_rmse_deg", "lower_rmse_deg",
                    "trunk_rmse_deg", "shoulder_flip_pct", "lag_frames", "jitter_deg",
                    "jerk_rms_deg_s3", "near_limit_pct", "violation_pct", "solve_ms_p50",
                    "solve_ms_p95", "solve_ms_max", "failed_frames")
    study_rows = {f.stem: read(f) for f in sorted((args.root / "studies").glob("*.csv"))}
    study_rows = {k: v for k, v in study_rows.items() if v}
    if study_rows:
        summary["studies"] = {}
        baseline = participant_means(study_rows.get("mhe_N7_o4c5", []), lambda r: float(r["joint_rmse_deg"]))
        for study, name in (("E3", "study_ik"), ("E4", "study_horizon"), ("E5", "study_filter")):
            variants = [v for v, rows in study_rows.items() if rows[0]["study"] in (study, "baseline")]
            order = {"E3": lambda v: study_rows[v][0]["ik_type"] != "mhe",
                     "E4": lambda v: int(study_rows[v][0]["N"]),
                     "E5": lambda v: float(study_rows[v][0]["filter_delay_1hz_ms"])}[study]
            variants.sort(key=order)
            table = []
            for v in variants:
                rows = study_rows[v]
                entry = {f: mean_sd(participant_means(rows, lambda r, f=f: float(r[f] or "nan")).values())
                         for f in study_fields}
                mine = participant_means(rows, lambda r: float(r["joint_rmse_deg"]))
                common = sorted(set(mine) & set(baseline))
                entry["rmse_minus_baseline_deg"] = mean_sd([mine[p] - baseline[p] for p in common])
                if len(common) >= 5 and v != "mhe_N7_o4c5":
                    from scipy import stats as st
                    diff = [mine[p] - baseline[p] for p in common]
                    entry["wilcoxon_p_vs_baseline"] = (1.0 if np.allclose(diff, 0)
                                                       else float(st.wilcoxon(diff).pvalue))
                entry["filter_delay_1hz_ms"] = float(rows[0]["filter_delay_1hz_ms"])
                summary["studies"].setdefault(study, {})[v] = entry
                table.append([v, rows[0]["ik_type"], rows[0]["N"], rows[0]["filter"],
                              fmt(entry["joint_rmse_deg"]), fmt(entry["rmse_minus_baseline_deg"]),
                              fmt(entry["joint_rmse_zero_lag_deg"]), fmt(entry["lag_frames"], 1),
                              f"{entry['filter_delay_1hz_ms']:.0f}", fmt(entry["jitter_deg"], 3),
                              fmt(entry["jerk_rms_deg_s3"], 0), fmt(entry["near_limit_pct"], 1),
                              fmt(entry["violation_pct"], 2), fmt(entry["solve_ms_p50"], 2),
                              fmt(entry["solve_ms_p95"], 2), fmt(entry["solve_ms_max"], 1),
                              fmt(entry["failed_frames"], 1), fmt(entry["shoulder_flip_pct"], 1),
                              f"{entry['wilcoxon_p_vs_baseline']:.4f}" if "wilcoxon_p_vs_baseline" in entry else "--"])
            write_table(tables / name,
                        ["variant", "IK", "N", "filter", "RMSE deg", "RMSE - baseline deg",
                         "RMSE at zero lag deg", "residual lag frames", "filter delay at 1 Hz ms",
                         "jitter deg/frame", "RMS jerk deg/s^3", "near limit %", "limit violation %",
                         "solve ms p50", "solve ms p95", "solve ms max", "failed frames",
                         "shoulder flip %", "Wilcoxon p vs baseline"], table)

    stat = stats_table("joint_rmse", by_arm_rmse, "deg")
    stats_table("reba_mae_own", by_arm_reba, "REBA points")
    stats_table("robot_body_rmse", by_arm_body, "mm", 1)
    stats_table("robot_hand_ee_rmse", by_arm_hands, "mm", 1)

    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"arms: {len(arms)} ({', '.join(arms)})" + (f"; missing: {missing}" if missing else ""))
    f = stat["friedman"]
    if f["chi2"] is not None:
        print(f"Friedman on whole-body RMSE over {len(f['arms'])} real-time arms, n = {f['n']}: "
              f"chi2({f['df']}) = {f['chi2']:.1f}, p = {f['p']:.2e}")
    print(f"tables written to {tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
