"""Hand-to-end-effector distance: signed errors, upper percentiles and missed close approaches.

Error = estimated distance - reference distance, so positive means the estimate
says the hand is farther than it is.
"""
import numpy as np

from rtcosmik.paper import scoring

#: 105 mm is the closest approach in the submitted paper (the "minimum separation" of R2-10).
THRESHOLDS_MM = (105, 150, 200, 300)
KINDS = ("causal", "aligned")

# Keep-or-drop rule of docs/tickets/13-robot-distance.md.
ABS_P99_LIMIT_MM = 105.0
OVER_P95_LIMIT_MM = 52.5
MISSED_LIMIT_PCT = 5.0
MIN_CLOSE_FRAMES = 100

#: Events (approaches, missed and false-alarm spells) last at least this long, and gaps
#: shorter than this do not split them: 125 ms at 40 Hz.
MIN_EVENT_FRAMES = 5
FRAME_MS = 25.0

COUNTS = ("close_frames", "approaches", "missed_frames", "missed_events", "undetected", "late",
          "false_alarm_frames", "false_alarm_events")
DELAYS = ("delay_median_frames", "delay_p95_frames", "delay_median_ms", "delay_p95_ms")
STATS = ("bias_mm", "sd_mm", "mae_mm", "rmse_mm", "abs_p95_mm", "abs_p99_mm", "abs_max_mm",
         "over_p95_mm", "over_p99_mm")
ROW_FIELDS = (["frames", "missing_frames", "clock_offset", "lag_frames", "ref_mean_mm",
               "ref_min_mm"]
              + [f"{k}_{s}" for k in KINDS for s in STATS]
              + [f"{k}_{t}mm_{c}" for k in KINDS for t in THRESHOLDS_MM for c in COUNTS + DELAYS])
POOLED_FIELDS = ROW_FIELDS[:2] + ROW_FIELDS[4:] + [f"{k}_{t}mm_{c}" for k in KINDS
                                                   for t in THRESHOLDS_MM
                                                   for c in ("missed_pct", "false_alarm_pct")]


def _on_reference(run, reference, lag):
    """``run`` by reference frame (run frame ``i + lag`` at ``i``), NaN where it has none."""
    out = np.full((len(reference),) + run.shape[1:], np.nan)
    paired, _ = scoring.shift(run, reference, lag)
    start = max(0, -lag)
    out[start:start + len(paired)] = paired
    return out


def _events(mask):
    """(start, stop) of the runs of True, short gaps bridged and short runs dropped."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    runs = []
    for start, stop in zip(edges[::2], edges[1::2]):
        if runs and start - runs[-1][1] < MIN_EVENT_FRAMES:
            runs[-1][1] = stop
        else:
            runs.append([start, stop])
    return [(a, b) for a, b in runs if b - a >= MIN_EVENT_FRAMES]


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


def approach_delays(estimate, reference, threshold):
    """Per approach of the reference below ``threshold``, the frames until the estimate is
    below it too: 0 on time, NaN never within the approach."""
    seen = estimate < threshold
    return np.array([np.argmax(seen[a:b]) if seen[a:b].any() else np.nan
                     for a, b in _events(reference < threshold)], float)


def delay_stats(delays):
    """Counts of approaches and the delay of those detected late."""
    late = delays[delays > 0]
    median, p95 = (np.median(late), np.percentile(late, 95)) if len(late) else (np.nan, np.nan)
    return {"approaches": len(delays), "undetected": int(np.isnan(delays).sum()),
            "late": len(late), "delay_median_frames": float(median),
            "delay_p95_frames": float(p95), "delay_median_ms": float(median * FRAME_MS),
            "delay_p95_ms": float(p95 * FRAME_MS)}


def threshold_counts(estimate, reference, threshold):
    """Reference frames below ``threshold`` and what the estimate made of them.

    A close frame without an estimate is missed; frames without a reference are not scored.
    Frame counts are raw, event counts follow ``_events``.
    """
    close, far = reference < threshold, reference >= threshold
    seen = estimate < threshold
    return {"close_frames": int(close.sum()), "missed_frames": int((close & ~seen).sum()),
            "missed_events": len(_events(close & ~seen)),
            "false_alarm_frames": int((seen & far).sum()),
            "false_alarm_events": len(_events(seen & far))}


def _scores(series):
    """Row of statistics and counts, and the approach delays per (kind, threshold)."""
    row, delays = {}, {}
    for kind in KINDS:
        row.update({f"{kind}_{k}": v
                    for k, v in error_stats(series[kind] - series["reference"]).items()})
        for t in THRESHOLDS_MM:
            delays[kind, t] = approach_delays(series[kind], series["reference"], t)
            counts = {**threshold_counts(series[kind], series["reference"], t),
                      **delay_stats(delays[kind, t])}
            row.update({f"{kind}_{t}mm_{k}": v for k, v in counts.items()})
    return row, delays


def score_hand(hand, reference_hand, effector, clock_offset, lag):
    """Score one hand of one trial. Returns a ``ROW_FIELDS`` dict and the per-frame distances.

    ``hand`` and ``effector`` are (F, 3) positions (m) by video frame, ``reference_hand`` by
    reference frame, NaN where missing. Video frame ``i + clock_offset`` is reference frame
    ``i``; ``lag`` is the shift that best aligns the estimate, clock offset included. The
    aligned estimate meets the robot where the reference does.
    """
    robot = _on_reference(effector, reference_hand, clock_offset)
    distance = lambda points: 1000.0 * np.linalg.norm(points - robot, axis=1)
    series = {"reference": distance(reference_hand),
              "causal": distance(_on_reference(hand, reference_hand, clock_offset)),
              "aligned": distance(_on_reference(hand, reference_hand, lag))}
    have = np.isfinite(series["reference"])
    scored = have & np.isfinite(series["causal"])
    if not scored.any():
        raise ValueError("no scored frame: estimate, reference and robot never coincide")
    ref = series["reference"][scored]
    row = {"frames": int(scored.sum()), "missing_frames": int((have & ~scored).sum()),
           "clock_offset": int(clock_offset), "lag_frames": int(lag - clock_offset),
           "ref_mean_mm": float(ref.mean()), "ref_min_mm": float(ref.min())}
    row.update(_scores(series)[0])
    return row, series


def pool(trials):
    """Pooled ``POOLED_FIELDS`` over the per-frame distances of several trials.

    Percentiles are over all frames and all approaches; counts are summed, so no run spans
    two trials.
    """
    rows, delays = zip(*[_scores(s) for s in trials])
    joined = {k: np.concatenate([s[k] for s in trials]) for k in ("reference",) + KINDS}
    have = np.isfinite(joined["reference"])
    scored = have & np.isfinite(joined["causal"])
    ref = joined["reference"][scored]
    out = {"frames": int(scored.sum()), "missing_frames": int((have & ~scored).sum()),
           "ref_mean_mm": float(ref.mean()), "ref_min_mm": float(ref.min())}
    for kind in KINDS:
        out.update({f"{kind}_{k}": v
                    for k, v in error_stats(joined[kind] - joined["reference"]).items()})
        for t in THRESHOLDS_MM:
            key = f"{kind}_{t}mm_"
            out.update({key + c: sum(r[key + c] for r in rows) for c in COUNTS})
            stats = delay_stats(np.concatenate([d[kind, t] for d in delays]))
            out.update({key + c: stats[c] for c in DELAYS})
            far = int(have.sum()) - out[key + "close_frames"]
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
