"""Paired statistics on participant means, reported by declared family only."""
from dataclasses import dataclass

import numpy as np
from scipy import stats

EQUIVALENCE_MARGIN_DEG = 1.0    # stated in advance; joint-angle errors only
ALPHA = 0.05
MIN_PARTICIPANTS = 5


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
    """Holm step-down adjusted p-values, in the order given."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def _signed_rank(d, alternative="two-sided"):
    return 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d, alternative=alternative).pvalue)


def paired_difference(diff):
    """Wilcoxon signed-rank p, mean difference with its t-based 95% CI, rank-biserial effect."""
    d = np.asarray(diff, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, len(d) - 1) * se
    nonzero = d[d != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    rbc = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()) if ranks.size else 0.0
    return {"n": len(d), "mean_diff": float(d.mean()),
            "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "p": _signed_rank(d), "rank_biserial": rbc}


def tost(diff, margin):
    """Two one-sided tests of |mean difference| < margin: paired t, and signed-rank beside it."""
    d = np.asarray(diff, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        p = 0.0 if abs(d.mean()) < margin else 1.0
    else:
        df = len(d) - 1
        p = max(stats.t.sf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))
    rank = max(_signed_rank(d + margin, "greater"), _signed_rank(d - margin, "less"))
    return {"p": float(p), "p_wilcoxon": rank}


def run_family(name, by_arm, margin=None):
    """Every comparison of a declared family, Holm-corrected over those the data allow.

    ``by_arm`` maps arm -> {participant: value}. With ``margin``, each comparison
    also gets its TOST p-values, Holm-corrected over the same family.
    """
    family = FAMILIES[name]
    rows, missing = [], []
    for a, b in family.comparisons:
        common = sorted(set(by_arm.get(a) or {}) & set(by_arm.get(b) or {}))
        if len(common) < MIN_PARTICIPANTS:
            missing.append((a, b))
            continue
        diff = np.array([by_arm[a][q] - by_arm[b][q] for q in common])
        row = {"a": a, "b": b, **paired_difference(diff)}
        if margin is not None:
            eq = tost(diff, margin)
            row.update(tost_p=eq["p"], tost_p_wilcoxon=eq["p_wilcoxon"])
        rows.append(row)
    for raw, adjusted in (("p", "p_holm"), ("tost_p", "tost_p_holm"),
                          ("tost_p_wilcoxon", "tost_p_wilcoxon_holm")):
        if rows and raw in rows[0]:
            for row, p in zip(rows, holm([r[raw] for r in rows])):
                row[adjusted] = float(p)
    for row in rows:
        row["significant"] = bool(row["p_holm"] < ALPHA)
        if margin is not None:
            row["equivalent"] = bool(row["tost_p_holm"] < ALPHA)
    return {"family": family.name, "title": family.title, "correction": "Holm",
            "declared": len(family.comparisons), "tested": len(rows), "margin": margin,
            "missing": missing, "comparisons": rows}


def omnibus(name, by_arm):
    """Friedman test over a declared family's arms, on the participants all of them cover."""
    arms = [a for a in dict.fromkeys(x for pair in FAMILIES[name].comparisons for x in pair)
            if by_arm.get(a)]
    common = sorted(set.intersection(*(set(by_arm[a]) for a in arms))) if arms else []
    if len(arms) < 3 or len(common) < MIN_PARTICIPANTS:
        return {"family": name, "arms": arms, "n": len(common), "chi2": None, "p": None, "df": None}
    chi2, p = stats.friedmanchisquare(*[[by_arm[a][q] for q in common] for a in arms])
    return {"family": name, "arms": arms, "n": len(common), "chi2": float(chi2), "p": float(p),
            "df": len(arms) - 1}
