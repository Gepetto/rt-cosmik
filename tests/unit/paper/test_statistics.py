"""Expected values are worked by hand or come from scipy calls the module does not make."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rtcosmik.paper import statistics as st


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


def test_holm_textbook_step_down():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # Holm step-down written the textbook way: p_(i) * (m - i + 1), cumulative max.
    order = np.argsort(p)
    expected = np.empty(len(p))
    expected[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    assert st.holm(p) == pytest.approx(expected)
    assert st.holm(p)[2] == pytest.approx(0.039 * 6)


def test_paired_difference_wilcoxon_exact_by_enumeration():
    # |d| ranks 1..5, negative rank sum 5: 10 of the 32 sign patterns give a sum <= 5.
    out = st.paired_difference([1.0, 2.0, 3.0, 4.0, -5.0])
    assert out["p"] == pytest.approx(20 / 32)
    assert out["n"] == 5
    assert out["mean_diff"] == pytest.approx(1.0)
    assert out["rank_biserial"] == pytest.approx((10 - 5) / 15)
    # all five positive: only the two extreme patterns are as extreme
    assert st.paired_difference([1.0, 2.0, 3.0, 4.0, 5.0])["p"] == pytest.approx(2 / 32)


def test_paired_difference_ci_hand_worked():
    # mean 0.5, sd 0.3, se 0.3 / sqrt(3); t(0.975, df = 2) = 4.302653 from tables
    out = st.paired_difference([0.5, 0.8, 0.2])
    half = 4.302653 * 0.3 / np.sqrt(3)
    assert out["ci95"] == pytest.approx([0.5 - half, 0.5 + half], abs=1e-5)


def test_paired_difference_all_zero():
    out = st.paired_difference([0.0] * 6)
    assert out["p"] == 1.0 and out["rank_biserial"] == 0.0 and out["ci95"] == [0.0, 0.0]


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


def test_tost_wilcoxon_variant_by_enumeration():
    # every d within the margin: d + 1 all positive, d - 1 all negative, no ties: 1 / 2^5 each
    out = st.tost([0.1, -0.2, 0.3, 0.05, -0.15], margin=1.0)
    assert out["p_wilcoxon"] == pytest.approx(1 / 32)
    # one d beyond the upper margin with the smallest |d - 1|: two patterns as extreme
    out = st.tost([1.01, -0.2, 0.3, 0.05, -0.15], margin=1.0)
    assert out["p_wilcoxon"] == pytest.approx(2 / 32)


def _arms(diffs, n=6):
    """A base arm and one arm per difference, each shifted by a small spread around it."""
    spread = np.array([0.03, -0.02, 0.01, -0.04, 0.025, 0.05])[:n]
    ids = [f"{1000 + i}" for i in range(n)]
    base = dict(zip(ids, np.linspace(8, 12, n)))
    arms = {"base": base}
    for name, d in diffs.items():
        arms[name] = {p: base[p] + d + s for p, s in zip(ids, spread)}
    return arms


@pytest.fixture
def family(monkeypatch):
    fam = st.Family("toy", "toy family", (("near", "base"), ("far", "base"), ("absent", "base")))
    monkeypatch.setitem(st.FAMILIES, "toy", fam)
    return fam


def test_run_family_corrects_over_the_comparisons_tested(family):
    out = st.run_family("toy", _arms({"near": 0.0, "far": 2.0}), margin=1.0)
    assert out["family"] == "toy" and out["declared"] == 3 and out["tested"] == 2
    near, far = out["comparisons"]
    assert (near["a"], far["a"]) == ("near", "far")
    assert far["p"] == pytest.approx(2 / 64)            # six positive differences, exact
    assert far["p_holm"] == pytest.approx(2 * far["p"])  # smallest of two
    assert near["p_holm"] == pytest.approx(max(near["p"], far["p_holm"]))
    assert near["tost_p_holm"] == pytest.approx(2 * near["tost_p"])
    assert near["equivalent"] and not far["equivalent"]
    assert far["significant"] is (far["p_holm"] < 0.05)
    assert out["missing"] == [("absent", "base")]


def test_run_family_without_margin_reports_no_equivalence(family):
    out = st.run_family("toy", _arms({"near": 0.0, "far": 2.0}))
    assert all("tost_p" not in c for c in out["comparisons"])


def test_run_family_uses_common_participants_only(family):
    arms = _arms({"near": 0.0, "far": 2.0})
    arms["near"]["9999"] = 50.0
    del arms["far"]["1000"]
    out = st.run_family("toy", arms, margin=1.0)
    assert [c["n"] for c in out["comparisons"]] == [6, 5]


def test_run_family_rejects_undeclared_family():
    with pytest.raises(KeyError):
        st.run_family("all pairs", {})


def test_omnibus_matches_scipy_friedman(family):
    from scipy import stats
    arms = _arms({"near": 0.0, "far": 2.0})
    out = st.omnibus("toy", arms)
    ids = sorted(arms["base"])
    chi2, p = stats.friedmanchisquare(*[[arms[a][q] for q in ids] for a in out["arms"]])
    assert sorted(out["arms"]) == ["base", "far", "near"]
    assert (out["chi2"], out["p"], out["df"], out["n"]) == pytest.approx((chi2, p, 2, 6))


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


def test_no_script_tests_outside_the_statistics_module():
    pattern = re.compile(r"wilcoxon\(|friedmanchisquare|ttest_|mannwhitneyu|kruskal\(|def holm")
    offenders = [str(f.relative_to(REPO)) for root in ("scripts", "src") for f in (REPO / root).rglob("*.py")
                 if f.name != "statistics.py" and pattern.search(f.read_text())]
    assert offenders == []
