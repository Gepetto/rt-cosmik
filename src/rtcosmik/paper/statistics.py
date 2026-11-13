"""Paired statistics on participant means, reported by declared family only."""
from dataclasses import dataclass

import numpy as np
from scipy import stats

EQUIVALENCE_MARGIN_DEG = 1.0    # stated in advance; joint-angle errors only
ALPHA = 0.05
MIN_PARTICIPANTS = 5
RANK_DECIMALS = 9               # differences are ranked after rounding: ties must not hinge on float noise


@dataclass(frozen=True)
class Family:
    name: str
    title: str
    comparisons: tuple      # (a, b): the difference tested is a - b


_PAPER_SETTING = "mhe_N7_o4c5"
FAMILIES = {f.name: f for f in (
    Family("arms", "proposed pipeline against its baselines, camera ladder and placement, FastSAM-3D at matched views", (
        ("nlf_0-2-4-6", "mmpose_0-2-4-6"), ("nlf_0-2-4-6", "nlf2d_0-2-4-6"),
        ("nlf_0", "nlf_0-2"), ("nlf_0-2", "nlf_0-2-4-6"), ("nlf_0", "nlf_0-2-4-6"),
        ("nlf_0-2", "nlf_0-4"),
        ("fastsam_0", "nlf_0"), ("fastsam_0-2", "nlf_0-2"), ("fastsam_0-4", "nlf_0-4"),
        ("fastsam_0-2-4-6", "nlf_0-2-4-6"))),
    Family("ik_filter", "IK and filter variants against the paper setting (MHE, N = 7, order 4 at 5 Hz)",
           tuple((v, _PAPER_SETTING) for v in (
               "sbs_o4c5",
               "mhe_N3_o4c5", "mhe_N5_o4c5", "mhe_N10_o4c5", "mhe_N15_o4c5", "mhe_N20_o4c5",
               "mhe_N7_nofilter", "mhe_N7_o2c6", "mhe_N7_o2c8", "mhe_N7_o2c10",
               "mhe_N7_o4c3", "mhe_N7_o4c4", "mhe_N7_o4c8", "mhe_N7_o4c10"))),
    Family("fusion", "fusion rules against inverse variance",
           tuple((v, "fusion_invvar") for v in ("fusion_mean", "fusion_reproj", "fusion_reproj_sym"))),
)}


def holm(pvalues):
    """Holm step-down adjusted p-values, in the order given.

    NaN is an untested member of the family: it counts in m as p = 1 and stays NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    untested = np.isnan(p)
    filled = np.where(untested, 1.0, p)
    order = np.argsort(filled, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(filled[order] * (len(p) - np.arange(len(p)))))
    adjusted[untested] = np.nan
    return adjusted


def _signed_rank(d, alternative="two-sided"):
    """Signed-rank p and the method used. Zero differences are dropped (Wilcoxon's rule);
    exact null distribution without ties or zeros, normal approximation otherwise."""
    d = np.round(d, RANK_DECIMALS)
    if not np.any(d):
        return 1.0, "none (all differences zero)"
    exact = np.all(d != 0) and len(np.unique(np.abs(d))) == len(d)
    p = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox", correction=False,
                       method="exact" if exact else "approx").pvalue
    return float(p), "exact" if exact else "normal approximation"


def _t_interval(d, level):
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - half), float(d.mean() + half)]


def paired_difference(diff):
    """Signed-rank p and its method, mean difference with unadjusted t intervals, rank-biserial effect."""
    d = np.asarray(diff, dtype=float)
    rounded = np.round(d, RANK_DECIMALS)
    nonzero = rounded[rounded != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    rbc = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()) if ranks.size else 0.0
    p, method = _signed_rank(d)
    return {"n": len(d), "mean_diff": float(d.mean()),
            "ci95": _t_interval(d, 0.95), "ci90": _t_interval(d, 0.90),
            "p": p, "wilcoxon_method": method, "rank_biserial": rbc}


def tost(diff, margin):
    """Two one-sided tests of |mean difference| < margin: paired t, and signed-rank beside it.

    With no spread the t statistic is undefined: p is 0 inside the margin and 1 otherwise.
    """
    d = np.asarray(diff, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        p = 0.0 if abs(d.mean()) < margin else 1.0
    else:
        df = len(d) - 1
        p = max(stats.t.sf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))
    rank = max(_signed_rank(d + margin, "greater")[0], _signed_rank(d - margin, "less")[0])
    return {"p": float(p), "p_wilcoxon": rank}


def matched_differences(a, b):
    """Participant mean of a - b over the trials both arms have, and the number of such trials.

    ``a`` and ``b`` map (participant, trial) -> value.
    """
    per_participant = {}
    for key in sorted(set(a) & set(b)):
        per_participant.setdefault(key[0], []).append(a[key] - b[key])
    return (np.array([np.mean(v) for v in per_participant.values()]),
            sum(len(v) for v in per_participant.values()))


def run_family(name, by_arm, margin=None):
    """Every comparison of a declared family, Holm-corrected over the family as declared.

    ``by_arm`` maps arm -> {(participant, trial): value}. A comparison the data do not
    allow is listed in ``missing`` and still counts in the correction, as p = 1.
    With ``margin``, each comparison also gets its TOST p-values, corrected the same way.
    """
    family = FAMILIES[name]
    slots, missing = [], []
    for a, b in family.comparisons:
        diff, trials = matched_differences(by_arm.get(a) or {}, by_arm.get(b) or {})
        if len(diff) < MIN_PARTICIPANTS:
            missing.append((a, b))
            slots.append(None)
            continue
        row = {"a": a, "b": b, **paired_difference(diff), "trials": trials}
        if margin is not None:
            eq = tost(diff, margin)
            row.update(tost_p=eq["p"], tost_p_wilcoxon=eq["p_wilcoxon"])
        slots.append(row)
    rows = [r for r in slots if r is not None]
    for raw, adjusted in (("p", "p_holm"), ("tost_p", "tost_p_holm"),
                          ("tost_p_wilcoxon", "tost_p_wilcoxon_holm")):
        if raw == "p" or margin is not None:
            corrected = holm([np.nan if r is None else r[raw] for r in slots])
            for row, p in zip(slots, corrected):
                if row is not None:
                    row[adjusted] = float(p)
    for row in rows:
        row["significant"] = bool(row["p_holm"] < ALPHA)
        if margin is not None:
            row["equivalent"] = bool(row["tost_p_holm"] < ALPHA)
    return {"family": family.name, "title": family.title, "correction": "Holm",
            "declared": len(family.comparisons), "tested": len(rows), "margin": margin,
            "missing": missing, "comparisons": rows}
