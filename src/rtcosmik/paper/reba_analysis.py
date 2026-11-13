"""Posture-only REBA of an arm against the reference: agreement, confusion, representative postures.

Every comparison exists for two pairings and three angle references:

``causal``   same frame after removing only the trial's clock offset
``aligned``  after the knee-flexion lag, as for the joint-angle errors

``absolute``           raw model angles for both series
``neutral_reference``  both relative to the reference's start-of-trial posture
``neutral_own``        each relative to its own start-of-trial posture
"""
import numpy as np

from rtcosmik.ergonomics import reba_posture as rp
from rtcosmik.paper import scoring

PAIRINGS = ("causal", "aligned")
ANGLE_REFERENCES = ("absolute", "neutral_reference", "neutral_own")
COMPONENTS = ("neck", "trunk", "legs", "upper_arm", "lower_arm")
LEVELS = len(rp.RISK_NAMES)
HIGH = rp.RISK_NAMES.index("high")
SCALARS = ("frames", "reba_est_mean", "reba_ref_mean", "reba_bias", "reba_mae", "reba_exact_pct",
           "risk_agree_pct", "risk_kappa", "high_frames", "high_under_pct", "high_missed_pct",
           "high_bias")


def _tables(run_dir):
    """A run as the column mappings ``reba_posture`` reads, indexed by video frame."""
    names, q, markers = scoring._by_frame(run_dir)
    joints = {name: q[:, i] for i, name in enumerate(names)}
    points = {f"{name}_{axis}": xyz[:, k] for name, xyz in markers.items()
              for k, axis in enumerate("xyz")}
    return names, q, joints, points


def _series(joints, points, neutral):
    out = rp.scores(joints, points, neutral)
    inputs = rp.corrected_inputs(joints, points, neutral)
    out["inputs"] = np.column_stack([inputs[key] for key in sorted(inputs)])
    return out


def paired_scores(run_dir, reference, clock_offset):
    """Per-frame REBA of a run and its reference, for every pairing and angle reference.

    Returns ``{(pairing, angles): {"frame", "est", "ref"}}`` and the run's lag in frames
    beyond the clock offset. ``frame`` is the reference frame of each pair.
    """
    names, q, joints, points = _tables(run_dir)
    ref_names, q_ref, ref_joints, ref_points = _tables(reference)
    if names != ref_names:
        raise ValueError(f"{run_dir}: joint columns differ from the reference")
    lag, _ = scoring._joint_lag(scoring.load_eval(), names, q, q_ref)

    start = lambda table: {key: values[rp.NEUTRAL_FRAMES] for key, values in table.items()}
    own = rp.neutral_from(start(joints), start(points))
    ref = rp.neutral_from(start(ref_joints), start(ref_points))
    neutrals = {"absolute": (None, None), "neutral_reference": (ref, ref),
                "neutral_own": (own, ref)}

    pairs = {}
    frames = np.arange(len(q_ref))
    for angles, (est_neutral, ref_neutral) in neutrals.items():
        est = _series(joints, points, est_neutral)
        truth = _series(ref_joints, ref_points, ref_neutral)
        for pairing, shift in (("causal", clock_offset), ("aligned", lag)):
            pair = {"est": {}, "ref": {}}
            for key in est:
                pair["est"][key], pair["ref"][key] = scoring._shift(est[key], truth[key], shift)
            pair["frame"] = frames[max(0, -shift):][:len(pair["ref"]["reba"])]
            pairs[pairing, angles] = pair
    return pairs, lag - clock_offset


def risk_levels(reba):
    return np.digitize(reba, rp.RISK_EDGES, right=True)


def weighted_kappa(confusion):
    """Linear-weighted Cohen's kappa of a confusion matrix of counts."""
    total = confusion.sum()
    if total == 0:
        return np.nan
    observed = confusion / total
    expected = np.outer(observed.sum(1), observed.sum(0))
    i, j = np.indices(confusion.shape)
    weights = np.abs(i - j) / (len(confusion) - 1)
    chance = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / chance) if chance > 0 else np.nan


def by_reference_level(confusion):
    """Confusion matrix as shares of each reference level (rows); NaN for unseen levels."""
    totals = confusion.sum(1, keepdims=True).astype(float)
    return confusion / np.where(totals > 0, totals, np.nan)


def agreement(est_reba, ref_reba):
    """Score and risk-level agreement over the frames both series scored.

    ``confusion[i, j]`` counts frames the reference rates level ``i`` and the estimate ``j``.
    The ``high_*`` figures cover frames the reference rates high or very high: the share the
    estimate rates at a lower level, the share it rates below high, and the mean signed error.
    """
    est_reba, ref_reba = np.asarray(est_reba, float), np.asarray(ref_reba, float)
    ok = np.isfinite(est_reba) & np.isfinite(ref_reba)
    e, r = est_reba[ok], ref_reba[ok]
    ke, kr = risk_levels(e), risk_levels(r)
    confusion = np.zeros((LEVELS, LEVELS), int)
    np.add.at(confusion, (kr, ke), 1)
    pct = lambda mask: float(100 * mask.mean()) if mask.size else np.nan
    mean = lambda x: float(x.mean()) if x.size else np.nan
    high = kr >= HIGH
    return {"frames": int(ok.sum()), "reba_est_mean": mean(e), "reba_ref_mean": mean(r),
            "reba_bias": mean(e - r), "reba_mae": mean(np.abs(e - r)),
            "reba_exact_pct": pct(e == r), "risk_agree_pct": pct(ke == kr),
            "risk_kappa": weighted_kappa(confusion), "confusion": confusion,
            "high_frames": int(high.sum()), "high_under_pct": pct(ke[high] < kr[high]),
            "high_missed_pct": pct(ke[high] < HIGH), "high_bias": mean((e - r)[high])}


def _longest_run(mask):
    """First and last index of the longest run of True, or None."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    starts, ends = edges[::2], edges[1::2]
    if not len(starts):
        return None
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k] - 1)


def representative_postures(pair, hold_frames):
    """Postures picked from the estimate alone, with what the reference says at those frames.

    ``peak``          middle of the longest spell at the highest REBA the estimate holds
                      for ``hold_frames``
    ``level_<name>``  per risk level the estimate visits, the frame closest to that
                      level's median posture; ``typical`` marks its most frequent level
    """
    est, ref = pair["est"], pair["ref"]
    reba = est["reba"]
    valid = np.isfinite(reba)
    if not valid.any():
        return []

    def row(selection, i, share, typical):
        out = {"selection": selection, "frame": int(pair["frame"][i]),
               "est_share_pct": float(share), "typical": bool(typical)}
        for name, series in (("est", est), ("ref", ref)):
            scored = np.isfinite(series["reba"][i])
            for key in ("reba", "risk", *COMPONENTS):
                out[f"{name}_{key}"] = float(series[key][i]) if scored else np.nan
        return out

    for score in np.unique(reba[valid])[::-1]:
        first, last = _longest_run(valid & (reba >= score))
        if last - first + 1 >= hold_frames:
            break
    rows = [row("peak", (first + last) // 2, 100 * np.mean(reba[valid] >= score), False)]

    shares = np.array([100 * np.mean(est["risk"][valid] == k) for k in range(LEVELS)])
    for level in np.flatnonzero(shares):
        at = np.flatnonzero(valid & (est["risk"] == level))
        posture = est["inputs"][at]
        nearest = at[np.argmin(np.linalg.norm(posture - np.median(posture, 0), axis=1))]
        rows.append(row(f"level_{rp.RISK_NAMES[level].replace(' ', '_')}", nearest,
                        shares[level], level == shares.argmax()))
    return rows


def mean_sd(values):
    """Mean, SD across participants (ddof 1) and how many had a value."""
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if not x.size:
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else np.nan, int(x.size)


def summarize(trials):
    """Pool trials ``(participant, est_reba, ref_reba)`` per participant, then across them.

    Returns the per-participant ``agreement``, ``{metric: (mean, sd, n)}`` over
    participants, and the agreement of all frames pooled.
    """
    grouped = {}
    for participant, est, ref in trials:
        grouped.setdefault(participant, []).append((est, ref))
    pool = lambda parts: agreement(np.concatenate([e for e, _ in parts]),
                                   np.concatenate([r for _, r in parts]))
    participants = {p: pool(parts) for p, parts in sorted(grouped.items())}
    across = {key: mean_sd([a[key] for a in participants.values()]) for key in SCALARS}
    pooled = pool([part for parts in grouped.values() for part in parts])
    return participants, across, pooled
