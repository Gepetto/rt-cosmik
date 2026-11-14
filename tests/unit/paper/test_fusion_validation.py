"""Expected values are literals worked by hand."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rtcosmik.paper import fusion_validation as fv

NAMES = ["C7", "RELB", "RWRI", "LELB", "LWRI", "RKNE", "RANK", "LKNE", "LANK"]


def _recording(frames=40000, views=4):
    rng = np.random.default_rng(0)
    poses = rng.normal(size=(frames, views, len(NAMES), 3))
    return rng.normal(size=(frames, views, len(NAMES), 2)), poses, np.ones((frames, views, len(NAMES)))


def _runs(flags):
    edges = np.flatnonzero(np.diff(np.concatenate([[False], flags, [False]]).astype(int)))
    return edges[1::2] - edges[0::2]


def test_stretches_cover_the_asked_share_in_runs_of_the_asked_length():
    flags = fv.stretches(200000, 0.25, 40, fv.seeded("a"))
    assert flags.mean() == pytest.approx(0.25, abs=0.02)
    assert _runs(flags).mean() == pytest.approx(40, rel=0.1)
    assert not fv.stretches(100, 0.0, 40, fv.seeded("a")).any()


def test_a_seed_depends_only_on_its_key():
    assert fv.seeded("1012", "Lifting").random() == fv.seeded("1012", "Lifting").random()
    assert fv.seeded("1012", "Lifting").random() != fv.seeded("1012", "Screwing").random()


def test_masked_views_lose_whole_frames_and_nothing_else():
    keypoints, poses, sigma = _recording()
    k, p, s = fv.mask_views(keypoints, poses, sigma, 0.3, 40, fv.seeded("m"))
    gone = np.isnan(p).all(axis=(2, 3))
    assert np.array_equal(gone, np.isnan(p).any(axis=(2, 3)))
    assert np.array_equal(gone, np.isnan(s).all(axis=2))
    assert np.array_equal(gone, np.isnan(k).all(axis=(2, 3)))
    assert np.array_equal(p[~gone], poses[~gone])
    assert gone.mean(axis=0) == pytest.approx([0.3] * 4, abs=0.04)
    assert np.isfinite(poses).all()


def test_outliers_move_one_limb_of_one_view_by_the_asked_distance():
    _, poses, _ = _recording()
    moved, view = fv.inject_outliers(poses, NAMES, 0.3, 0.2, 40, fv.seeded("o"))
    distance = np.linalg.norm(moved - poses, axis=3)            # (F, C, M)
    others = [c for c in range(4) if c != view]
    assert not distance[:, others].any()
    assert not distance[:, view, NAMES.index("C7")].any()
    hit = distance[:, view] > 0
    np.testing.assert_allclose(distance[:, view][hit], 0.3)
    assert hit.any(axis=1).mean() == pytest.approx(0.2, abs=0.04)
    limbs = {tuple(sorted(np.flatnonzero(row))) for row in hit[hit.any(axis=1)]}
    assert limbs == {(1, 2), (3, 4), (5, 6), (7, 8)}


def test_rank_correlation_of_monotone_and_reversed_series():
    a = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    b = np.stack([a ** 3, -a, [3.0, 3.0, 3.0, 3.0, 3.0], [1.0, 3.0, 2.0, 5.0, 4.0]], axis=1)
    rho = fv.rank_correlation(a[:, None], b)
    # last column: rank differences 0, 1, 1, 1, 1 -> 1 - 6*4/(5*24) = 0.8
    np.testing.assert_allclose(rho[:3], [1.0, -1.0, np.nan])
    assert rho[3] == pytest.approx(0.8)


def test_tied_values_share_their_average_rank():
    np.testing.assert_allclose(fv.ranks(np.array([10.0, 30.0, 10.0, 20.0, 30.0, 30.0])),
                               [1.5, 5.0, 1.5, 3.0, 5.0, 5.0])


def test_calibration_bins_errors_by_sigma():
    sigma = np.array([0.01, 0.02, 0.05, 0.06, 0.07])
    error = np.array([0.03, 0.04, 0.01, 0.02, 0.09])
    low, high = fv.calibration(sigma, error, [0.0, 0.03, 0.1])
    assert (low["count"], high["count"]) == (2, 3)
    assert low["sigma_mean"] == pytest.approx(0.015)
    assert low["error_rms"] == pytest.approx(np.sqrt(0.00125))
    assert high["error_mean"] == pytest.approx(0.04)


def test_view_choice_counts_frames_where_sigma_picks_the_right_view():
    sigma = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 1.0, 3.0]])
    error = np.array([[5.0, 6.0, 7.0], [9.0, 6.0, 7.0], [9.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    best, worst = fv.view_choice(sigma, error)
    assert (best, worst) == (0.25, 0.75)
