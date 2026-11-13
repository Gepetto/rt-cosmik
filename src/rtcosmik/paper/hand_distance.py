"""Hand-to-end-effector distance: signed errors, upper percentiles and missed close approaches.

Error = estimated distance - reference distance, so positive means the estimate
says the hand is farther than it is.
"""
import numpy as np

#: 105 mm is the closest approach in the submitted paper (the "minimum separation" of R2-10).
THRESHOLDS_MM = (105, 150, 200, 300)
KINDS = ("causal", "aligned")

# Keep-or-drop rule of docs/tickets/13-robot-distance.md.
ABS_P99_LIMIT_MM = 105.0
OVER_P95_LIMIT_MM = 52.5
MISSED_LIMIT_PCT = 5.0
MIN_CLOSE_FRAMES = 100

COUNTS = ("close_frames", "approaches", "missed_frames", "missed_events", "undetected",
          "false_alarm_frames", "false_alarm_events")
STATS = ("bias_mm", "sd_mm", "mae_mm", "rmse_mm", "abs_p95_mm", "abs_p99_mm", "abs_max_mm",
         "over_p95_mm", "over_p99_mm")
ROW_FIELDS = (["frames", "clock_offset", "lag_frames", "ref_mean_mm", "ref_min_mm"]
              + [f"{k}_{s}" for k in KINDS for s in STATS]
              + [f"{k}_{t}mm_{c}" for k in KINDS for t in THRESHOLDS_MM for c in COUNTS])
POOLED_FIELDS = ROW_FIELDS[:1] + ROW_FIELDS[3:] + [f"{k}_{t}mm_{c}" for k in KINDS
                                                   for t in THRESHOLDS_MM
                                                   for c in ("missed_pct", "false_alarm_pct")]


def _at(series, index):
    """Rows of ``series`` at ``index``, NaN where the index falls outside."""
    out = np.full((len(index),) + series.shape[1:], np.nan)
    ok = (index >= 0) & (index < len(series))
    out[ok] = series[index[ok]]
    return out


def _runs(mask):
    """(start, stop) of every contiguous run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def error_stats(error):
    """Signed-error statistics (mm) of finite ``error`` values."""
    e = error[np.isfinite(error)]
    if not len(e):
        return dict.fromkeys(STATS, np.nan)
    p = lambda x, q: float(np.percentile(x, q))
    return {"bias_mm": float(e.mean()), "sd_mm": float(e.std()), "mae_mm": float(np.abs(e).mean()),
            "rmse_mm": float(np.sqrt((e ** 2).mean())),
            "abs_p95_mm": p(np.abs(e), 95), "abs_p99_mm": p(np.abs(e), 99),
            "abs_max_mm": float(np.abs(e).max()),
            "over_p95_mm": p(e, 95), "over_p99_mm": p(e, 99)}


def threshold_counts(estimate, reference, threshold):
    """Close approaches of the reference below ``threshold`` and what the estimate made of them.

    Frames where either distance is missing are not scored and end a run.
    """
    ok = np.isfinite(estimate) & np.isfinite(reference)
    close = ok & (reference < threshold)
    seen = ok & (estimate < threshold)
    approaches = _runs(close)
    return {"close_frames": int(close.sum()), "approaches": len(approaches),
            "missed_frames": int((close & ~seen).sum()), "missed_events": len(_runs(close & ~seen)),
            "undetected": sum(not seen[a:b].any() for a, b in approaches),
            "false_alarm_frames": int((seen & ~close).sum()),
            "false_alarm_events": len(_runs(seen & ~close))}


def _scores(series):
    row = {}
    for kind in KINDS:
        row.update({f"{kind}_{k}": v
                    for k, v in error_stats(series[kind] - series["reference"]).items()})
        for t in THRESHOLDS_MM:
            counts = threshold_counts(series[kind], series["reference"], t)
            row.update({f"{kind}_{t}mm_{k}": v for k, v in counts.items()})
    return row


def score_hand(hand, reference_hand, effector, clock_offset, lag):
    """Score one hand of one trial. Returns a ``ROW_FIELDS`` dict and the per-frame distances.

    ``hand`` and ``effector`` are (F, 3) positions (m) by video frame, ``reference_hand`` by
    reference frame, NaN where missing. Video frame ``i + clock_offset`` is reference frame
    ``i``; ``lag`` is the shift that best aligns the estimate, clock offset included. The
    aligned estimate meets the robot where the reference does.
    """
    i = np.arange(len(reference_hand))
    robot = _at(effector, i + clock_offset)
    distance = lambda points: 1000.0 * np.linalg.norm(points - robot, axis=1)
    series = {"reference": distance(reference_hand),
              "causal": distance(_at(hand, i + clock_offset)),
              "aligned": distance(_at(hand, i + lag))}
    scored = np.isfinite(series["reference"]) & np.isfinite(series["causal"])
    ref = series["reference"][scored]
    row = {"frames": int(scored.sum()), "clock_offset": int(clock_offset),
           "lag_frames": int(lag - clock_offset),
           "ref_mean_mm": float(ref.mean()) if len(ref) else np.nan,
           "ref_min_mm": float(ref.min()) if len(ref) else np.nan}
    row.update(_scores(series))
    return row, series


def pool(trials):
    """Pooled ``POOLED_FIELDS`` over the per-frame distances of several trials.

    Percentiles are over all frames; counts are summed, so no run spans two trials.
    """
    rows = [_scores(s) for s in trials]
    joined = {k: np.concatenate([s[k] for s in trials]) for k in ("reference",) + KINDS}
    scored = np.isfinite(joined["reference"]) & np.isfinite(joined["causal"])
    ref = joined["reference"][scored]
    out = {"frames": int(scored.sum()), "ref_mean_mm": float(ref.mean()),
           "ref_min_mm": float(ref.min())}
    for kind in KINDS:
        out.update({f"{kind}_{k}": v
                    for k, v in error_stats(joined[kind] - joined["reference"]).items()})
        for t in THRESHOLDS_MM:
            key = f"{kind}_{t}mm_"
            out.update({key + c: sum(r[key + c] for r in rows) for c in COUNTS})
            ok = np.isfinite(joined[kind]) & np.isfinite(joined["reference"])
            far = int(ok.sum()) - out[key + "close_frames"]
            ratio = lambda a, b: 100.0 * a / b if b else np.nan
            out[key + "missed_pct"] = ratio(out[key + "missed_frames"], out[key + "close_frames"])
            out[key + "false_alarm_pct"] = ratio(out[key + "false_alarm_frames"], far)
    return out


def decide(pooled):
    """The keep-or-drop rule on one hand's pooled causal row."""
    evaluable = [t for t in THRESHOLDS_MM
                 if pooled[f"causal_{t}mm_close_frames"] >= MIN_CLOSE_FRAMES]
    threshold = evaluable[0] if evaluable else None
    missed = pooled[f"causal_{threshold}mm_missed_pct"] if evaluable else np.nan
    checks = {"abs_p99": pooled["causal_abs_p99_mm"] < ABS_P99_LIMIT_MM,
              "over_p95": pooled["causal_over_p95_mm"] < OVER_P95_LIMIT_MM,
              "missed": bool(missed < MISSED_LIMIT_PCT)}
    return {"keep": all(checks.values()), "missed_threshold_mm": threshold,
            "missed_pct": missed, **{f"{k}_ok": bool(v) for k, v in checks.items()}}
