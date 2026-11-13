"""Expected values are literals worked by hand, closed forms or enumerations; one TOST check
uses scipy's one-sample t-test, which the module does not call."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

from rtcosmik.paper import statistics as st
import aggregate_results as agg


def t2_cdf(t):
    """Student t CDF with 2 degrees of freedom, closed form."""
    return 0.5 + t / (2 * np.sqrt(2 + t * t))


def test_holm_hand_worked():
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> monotone 0.03, 0.06, 0.06
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order():
    assert st.holm([0.5, 0.9, 0.001]) == pytest.approx([1.0, 1.0, 0.003])
    assert st.holm([0.02]) == pytest.approx([0.02])
    assert len(st.holm([])) == 0


def test_holm_eight_values_hand_worked():
    # sorted: .001 .008 .039 .041 .042 .06 .074 .205, times 8..1:
    # .008 .056 .234 .205 .168 .18 .148 .205 -> running maximum .008 .056 then .234
    p = [0.041, 0.001, 0.205, 0.039, 0.06, 0.008, 0.074, 0.042]
    assert st.holm(p) == pytest.approx([0.234, 0.008, 0.234, 0.234, 0.234, 0.056, 0.234, 0.234])


def test_holm_nan_is_an_untested_member_of_the_family():
    # m = 3: 0.01 x 3, 0.03 x 2; the untested one counts as p = 1 and stays NaN
    out = st.holm([0.01, float("nan"), 0.03])
    assert out[[0, 2]] == pytest.approx([0.03, 0.06])
    assert np.isnan(out[1])


def test_paired_difference_wilcoxon_exact_by_enumeration():
    # |d| ranks 1..5, negative rank sum 5: 10 of the 32 sign patterns give a sum <= 5.
    out = st.paired_difference([1.0, 2.0, 3.0, 4.0, -5.0])
    assert out["p"] == pytest.approx(20 / 32)
    assert out["n"] == 5
    assert out["mean_diff"] == pytest.approx(1.0)
    assert out["rank_biserial"] == pytest.approx((10 - 5) / 15)
    # all five positive: only the two extreme patterns are as extreme
    assert st.paired_difference([1.0, 2.0, 3.0, 4.0, 5.0])["p"] == pytest.approx(2 / 32)
    assert out["wilcoxon_method"] == "exact"


def test_paired_difference_ties_use_the_normal_approximation():
    # ranks 1.5 1.5 3 4 5, W- = 5, mean 7.5, variance 13.75 - (2^3 - 2) / 48
    out = st.paired_difference([1.0, 1.0, 2.0, 3.0, -4.0])
    assert out["wilcoxon_method"] == "normal approximation"
    assert out["p"] == pytest.approx(math.erfc(2.5 / math.sqrt(13.625) / math.sqrt(2)))


def test_ties_do_not_depend_on_floating_point_noise():
    # 0.1 + 0.2 and 0.3 differ in the last bit: still a tie, same p as the tied case above
    noisy = st.paired_difference([0.1 + 0.2, 0.3, 0.6, 0.9, -1.2])
    assert noisy["wilcoxon_method"] == "normal approximation"
    assert noisy["p"] == pytest.approx(math.erfc(2.5 / math.sqrt(13.625) / math.sqrt(2)))
    assert noisy["rank_biserial"] == pytest.approx((10 - 5) / 15)
    assert st.paired_difference([0.1 + 0.2 - 0.3, 1.0, 2.0, 3.0, -4.0, 5.0])["wilcoxon_method"] != "exact"


def test_paired_difference_zeros_are_dropped_and_approximated():
    # the zero is dropped: n = 5, ranks 1..5, W- = 4, mean 7.5, variance 13.75
    out = st.paired_difference([0.0, 1.0, 2.0, 3.0, -4.0, 5.0])
    assert out["wilcoxon_method"] == "normal approximation"
    assert out["p"] == pytest.approx(math.erfc(3.5 / math.sqrt(13.75) / math.sqrt(2)))
    assert out["n"] == 6


def test_paired_difference_ci_hand_worked():
    # mean 0.5, sd 0.3, se 0.3 / sqrt(3); t(0.975, df = 2) = 4.302653 from tables
    out = st.paired_difference([0.5, 0.8, 0.2])
    half = 4.302653 * 0.3 / np.sqrt(3)
    assert out["ci95"] == pytest.approx([0.5 - half, 0.5 + half], abs=1e-5)
    half90 = 2.919986 * 0.3 / np.sqrt(3)    # t(0.95, df = 2)
    assert out["ci90"] == pytest.approx([0.5 - half90, 0.5 + half90], abs=1e-5)


def test_paired_difference_all_zero():
    out = st.paired_difference([0.0] * 6)
    assert out["p"] == 1.0 and out["rank_biserial"] == 0.0 and out["ci95"] == [0.0, 0.0]
    assert out["wilcoxon_method"] == "none (all differences zero)"


def test_tost_hand_worked_symmetric():
    # mean 0, se 0.3 / sqrt(3): both one-sided t are 1 / se, p from the df = 2 closed form
    out = st.tost([0.0, 0.3, -0.3], margin=1.0)
    expected = 1 - t2_cdf(1.0 / (0.3 / np.sqrt(3)))
    assert expected == pytest.approx(0.014357, abs=1e-6)
    assert out["p"] == pytest.approx(expected)


def test_tost_hand_worked_takes_the_worse_side():
    # mean 0.5: the upper bound is the binding one, t = (0.5 - 1) / se
    out = st.tost([0.5, 0.8, 0.2], margin=1.0)
    assert out["p"] == pytest.approx(t2_cdf(-0.5 / (0.3 / np.sqrt(3))))
    assert out["p"] == pytest.approx(0.050987, abs=1e-6)


def test_tost_matches_scipy_one_sided_t_tests():
    from scipy import stats
    rng = np.random.default_rng(3)
    d = rng.normal(0.3, 0.6, 18)
    lower = stats.ttest_1samp(d, -1.0, alternative="greater").pvalue
    upper = stats.ttest_1samp(d, 1.0, alternative="less").pvalue
    assert st.tost(d, margin=1.0)["p"] == pytest.approx(max(lower, upper))


def test_tost_fails_outside_the_margin_and_on_wide_spread():
    assert st.tost([1.4, 1.5, 1.6, 1.5, 1.45], margin=1.0)["p"] > 0.95
    assert st.tost([-3.0, 3.0, -2.5, 2.5, 0.0], margin=1.0)["p"] > 0.05


def test_tost_zero_spread():
    assert st.tost([0.2] * 5, margin=1.0)["p"] == 0.0
    assert st.tost([1.2] * 5, margin=1.0)["p"] == 1.0


def test_exact_zero_p_is_printed_as_zero():
    assert agg.fmt_p(0.0) == "0"
    assert agg.fmt_p(3.4e-34) == "3.4e-34" and agg.fmt_p(0.0312) == "0.0312"


def test_tost_wilcoxon_variant_by_enumeration():
    # every d within the margin: d + 1 all positive, d - 1 all negative, no ties: 1 / 2^5 each
    out = st.tost([0.1, -0.2, 0.3, 0.05, -0.15], margin=1.0)
    assert out["p_wilcoxon"] == pytest.approx(1 / 32)
    # one d beyond the upper margin with the smallest |d - 1|: two patterns as extreme
    out = st.tost([1.01, -0.2, 0.3, 0.05, -0.15], margin=1.0)
    assert out["p_wilcoxon"] == pytest.approx(2 / 32)


TASKS = ("T1", "T2")


def _arms(diffs, n=6):
    """A base arm and one arm per difference, per (participant, task), with a small spread."""
    spread = np.array([0.03, -0.02, 0.01, -0.04, 0.025, 0.05])[:n]
    ids = [f"{1000 + i}" for i in range(n)]
    base = {(p, t): v + 10 * k for p, v in zip(ids, np.linspace(8, 12, n)) for k, t in enumerate(TASKS)}
    arms = {"base": base}
    for name, d in diffs.items():
        arms[name] = {(p, t): base[p, t] + d + s for p, s in zip(ids, spread) for t in TASKS}
    return arms


@pytest.fixture
def family(monkeypatch):
    fam = st.Family("toy", "toy family", (("near", "base"), ("far", "base"), ("absent", "base")))
    monkeypatch.setitem(st.FAMILIES, "toy", fam)
    return fam


def test_run_family_corrects_over_the_declared_family(family):
    out = st.run_family("toy", _arms({"near": 0.0, "far": 2.0}), margin=1.0)
    assert out["family"] == "toy" and out["declared"] == 3 and out["tested"] == 2
    near, far = out["comparisons"]
    assert (near["a"], far["a"]) == ("near", "far")
    assert far["p"] == pytest.approx(2 / 64)            # six positive differences, exact
    assert far["p_holm"] == pytest.approx(3 * 2 / 64)   # smallest of the three declared
    assert near["p_holm"] == pytest.approx(min(1.0, max(2 * near["p"], far["p_holm"])))
    assert near["tost_p_holm"] == pytest.approx(3 * near["tost_p"])
    assert near["equivalent"] and not far["equivalent"]
    assert not far["significant"]                       # 0.094: would pass with m = 2
    assert out["missing"] == [("absent", "base")]


def test_running_fewer_arms_never_weakens_the_correction(family):
    arms = _arms({"near": 0.0, "far": 2.0})
    full = st.run_family("toy", arms, margin=1.0)["comparisons"][1]
    del arms["near"]
    alone, = st.run_family("toy", arms, margin=1.0)["comparisons"]
    assert alone["p_holm"] == pytest.approx(3 * alone["p"])
    assert alone["p_holm"] >= full["p_holm"] and alone["tost_p_holm"] >= full["tost_p_holm"]


def test_run_family_without_margin_reports_no_equivalence(family):
    out = st.run_family("toy", _arms({"near": 0.0, "far": 2.0}))
    assert all("tost_p" not in c and "equivalent" not in c for c in out["comparisons"])


def test_run_family_pairs_the_trials_both_arms_have(family):
    arms = _arms({"near": 0.0, "far": 2.0})
    # T2 is 10 deg above T1 in every arm: an unmatched mean would move this participant by 5 deg
    del arms["base"]["1000", "T2"]
    out = st.run_family("toy", arms, margin=1.0)
    near, far = out["comparisons"]
    assert (near["n"], near["trials"]) == (6, 11)
    assert near["mean_diff"] == pytest.approx(np.mean([0.03, -0.02, 0.01, -0.04, 0.025, 0.05]))
    assert far["mean_diff"] == pytest.approx(2 + near["mean_diff"])


def test_run_family_drops_participants_without_a_common_trial(family):
    arms = _arms({"near": 0.0, "far": 2.0})
    arms["near"]["9999", "T1"] = 50.0
    del arms["far"]["1001", "T1"], arms["base"]["1001", "T2"]
    out = st.run_family("toy", arms, margin=1.0)
    assert [(c["n"], c["trials"]) for c in out["comparisons"]] == [(6, 11), (5, 10)]


def test_run_family_rejects_undeclared_family():
    with pytest.raises(KeyError):
        st.run_family("all pairs", {})


def test_no_omnibus_test_in_the_module():
    assert not hasattr(st, "omnibus")


def test_family_table_lists_untested_comparisons_and_states_the_method(family, tmp_path):
    arms = _arms({"near": 0.0, "far": 2.0})
    arms["near"] = {k: arms["base"][k] + 0.25 for k in arms["base"]}     # zero spread: TOST p = 0
    out = st.run_family("toy", arms, margin=1.0)
    agg.write_family(tmp_path / "stats_toy", [out], "deg")
    text = (tmp_path / "stats_toy.md").read_text()
    lines = text.splitlines()
    absent = next(line for line in lines if "| absent |" in line)
    assert "not tested" in absent
    assert "Holm over the 3 comparisons declared, 2 tested" in text
    assert "unadjusted 95% t-intervals on participant mean differences" in text
    assert "90% interval lying inside the margin" in text
    assert "A p-value printed as 0" in text
    near = next(line for line in lines if "| near |" in line).split(" | ")
    header = lines[0].split(" | ")
    assert near[header.index("TOST p")] == "0" and near[header.index("TOST p Holm")] == "0"
    assert near[header.index("participants")] == "6" and near[header.index("common trials")] == "12"
    assert "exact" in lines[3].split(" | ")[header.index("signed-rank method")]


def test_declared_families():
    assert st.EQUIVALENCE_MARGIN_DEG == 1.0
    arms = st.FAMILIES["arms"].comparisons
    assert ("fastsam_0", "nlf_0") in arms and ("nlf_0-2-4-6", "mmpose_0-2-4-6") in arms
    ik = st.FAMILIES["ik_filter"].comparisons
    assert all(b == "mhe_N7_o4c5" for _, b in ik)
    assert {"sbs_o4c5", "mhe_N3_o4c5", "mhe_N20_o4c5", "mhe_N7_nofilter"} <= {a for a, _ in ik}
    fusion = st.FAMILIES["fusion"].comparisons
    assert {a for a, _ in fusion} == {"fusion_mean", "fusion_reproj", "fusion_reproj_sym"}
    assert all(b == "fusion_invvar" for _, b in fusion)
    for fam in st.FAMILIES.values():
        assert len(set(fam.comparisons)) == len(fam.comparisons)


STATISTICS_MODULE = "src/rtcosmik/paper/statistics.py"
#: Non-test uses of scipy.stats outside the module: path -> the call it is allowed to make.
SCIPY_STATS_ALLOWED = {}


def test_no_statistical_test_outside_the_statistics_module():
    tests = ("pearsonr|spearmanr|linregress|permutation_test|ttest\\w*|mannwhitneyu|kruskal|wilcoxon|"
             "friedmanchisquare|f_oneway|chi2_contingency")
    pattern = re.compile(rf"\b(?:{tests})\s*\(|\bdef holm\b|\bscipy\.stats\b[.\w]*|"
                         r"from\s+scipy\s+import\s+[^\n]*\bstats\b")
    offenders = {}
    for root in ("scripts", "src"):
        for f in (REPO / root).rglob("*.py"):
            path = f.relative_to(REPO).as_posix()
            hits = [h for h in pattern.findall(f.read_text()) if h not in SCIPY_STATS_ALLOWED.get(path, ())]
            if path != STATISTICS_MODULE and hits:
                offenders[path] = hits
    assert offenders == {}
