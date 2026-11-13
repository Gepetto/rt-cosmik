"""Hand-to-end-effector distance statistics on synthetic trajectories."""
import numpy as np
import pytest

from rtcosmik.paper import hand_distance as hd

FRAMES = 400


def _along_x(distance_m):
    return np.stack([distance_m, np.zeros_like(distance_m), np.zeros_like(distance_m)], 1)


def _still_effector():
    return np.zeros((FRAMES, 3))


def test_constant_offset_is_the_signed_error():
    reference = _along_x(np.linspace(0.2, 0.5, FRAMES))
    estimate = reference + [0.03, 0.0, 0.0]                  # 30 mm farther than it is
    row, _ = hd.score_hand(estimate, reference, _still_effector(), clock_offset=0, lag=0)
    for kind in ("causal", "aligned"):
        assert row[f"{kind}_bias_mm"] == pytest.approx(30.0)
        assert row[f"{kind}_sd_mm"] == pytest.approx(0.0, abs=1e-9)
        assert row[f"{kind}_abs_p95_mm"] == pytest.approx(30.0)
        assert row[f"{kind}_abs_p99_mm"] == pytest.approx(30.0)
        assert row[f"{kind}_over_p95_mm"] == pytest.approx(30.0)


def test_a_closer_estimate_has_a_negative_error():
    reference = _along_x(np.full(FRAMES, 0.3))
    row, _ = hd.score_hand(reference - [0.02, 0.0, 0.0], reference, _still_effector(), 0, 0)
    assert row["causal_bias_mm"] == pytest.approx(-20.0)
    assert row["causal_abs_p99_mm"] == pytest.approx(20.0)


def test_lag_counts_in_causal_error_and_not_in_aligned_error():
    clock, pipeline = 2, 5
    reference = _along_x(0.2 + 0.001 * np.arange(FRAMES))     # recedes 1 mm per frame
    estimate = np.full((FRAMES, 3), np.nan)
    estimate[clock + pipeline:] = reference[:FRAMES - clock - pipeline]
    row, _ = hd.score_hand(estimate, reference, _still_effector(), clock, clock + pipeline)
    assert row["lag_frames"] == pipeline
    assert row["causal_bias_mm"] == pytest.approx(-5.0)       # 5 frames late on a 1 mm/frame ramp
    assert row["causal_abs_p99_mm"] == pytest.approx(5.0)
    assert row["aligned_bias_mm"] == pytest.approx(0.0, abs=1e-9)
    assert row["aligned_abs_p99_mm"] == pytest.approx(0.0, abs=1e-9)


def test_clock_offset_alone_is_not_an_error():
    """Video and robot states trail the reference by the clock offset; the robot moves."""
    clock = 3
    effector_true = _along_x(0.002 * np.arange(FRAMES))       # by reference frame
    reference = effector_true + [0.0, 0.25, 0.0]
    effector = np.full((FRAMES, 3), np.nan)                   # by video frame
    effector[clock:] = effector_true[:FRAMES - clock]
    estimate = np.full((FRAMES, 3), np.nan)
    estimate[clock:] = reference[:FRAMES - clock]
    row, _ = hd.score_hand(estimate, reference, effector, clock, clock)
    assert row["frames"] == FRAMES - clock
    assert row["ref_mean_mm"] == pytest.approx(250.0)
    assert row["causal_abs_p99_mm"] == pytest.approx(0.0, abs=1e-9)


def _dips(*spans, base=0.3):
    d = np.full(FRAMES, base)
    for start, stop, value in spans:
        d[start:stop] = value
    return _along_x(d)


def test_a_missed_close_approach_is_one_event():
    reference = _dips((40, 60, 0.100))                        # 20 frames below 105 mm
    estimate = _dips()                                        # never sees it
    row, _ = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_close_frames"] == 20
    assert row["causal_105mm_approaches"] == 1
    assert row["causal_105mm_missed_frames"] == 20
    assert row["causal_105mm_missed_events"] == 1
    assert row["causal_105mm_undetected"] == 1
    assert row["causal_105mm_false_alarm_frames"] == 0
    assert row["causal_105mm_false_alarm_events"] == 0
    # 100 mm is also below the wider thresholds; the estimate at 300 mm is not below 300.
    assert row["causal_300mm_missed_events"] == 1


def test_a_late_detection_is_a_missed_event_but_not_an_undetected_approach():
    reference = _dips((40, 60, 0.100), (200, 230, 0.090))
    estimate = _dips((45, 60, 0.100), (200, 230, 0.090))      # first one seen 5 frames late
    row, _ = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_close_frames"] == 50
    assert row["causal_105mm_approaches"] == 2
    assert row["causal_105mm_missed_frames"] == 5
    assert row["causal_105mm_missed_events"] == 1
    assert row["causal_105mm_undetected"] == 0


def test_an_estimate_below_the_threshold_alone_is_a_false_alarm():
    reference = _dips()
    estimate = _dips((100, 110, 0.090), (300, 304, 0.100))
    row, _ = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_false_alarm_frames"] == 14
    assert row["causal_105mm_false_alarm_events"] == 2
    assert row["causal_105mm_missed_frames"] == 0
    assert row["causal_105mm_approaches"] == 0


def test_the_stated_thresholds_include_105_mm():
    assert 105 in hd.THRESHOLDS_MM


def test_pooling_trials_adds_counts_and_takes_percentiles_over_all_frames():
    quiet = hd.score_hand(_dips() + [0.01, 0, 0], _dips(), _still_effector(), 0, 0)[1]
    missed = hd.score_hand(_dips(), _dips((40, 60, 0.100)), _still_effector(), 0, 0)[1]
    pooled = hd.pool([quiet, missed])
    assert pooled["frames"] == 2 * FRAMES
    assert pooled["causal_105mm_missed_events"] == 1
    assert pooled["causal_105mm_missed_pct"] == pytest.approx(100.0)
    # errors: 400 frames at +10 mm, 380 at 0, 20 at +200 mm
    assert pooled["causal_bias_mm"] == pytest.approx((400 * 10 + 20 * 200) / 800)
    assert pooled["causal_abs_p99_mm"] == pytest.approx(200.0)
    assert pooled["causal_abs_p95_mm"] == pytest.approx(10.0)


def _pooled(p99, over_p95, close, missed_pct, close_150=0, missed_150=0.0):
    row = {"causal_abs_p99_mm": p99, "causal_over_p95_mm": over_p95}
    for t in hd.THRESHOLDS_MM:
        row[f"causal_{t}mm_close_frames"], row[f"causal_{t}mm_missed_pct"] = 0, float("nan")
    row["causal_105mm_close_frames"], row["causal_105mm_missed_pct"] = close, missed_pct
    row["causal_150mm_close_frames"], row["causal_150mm_missed_pct"] = close_150, missed_150
    return row


def test_decision_rule():
    assert hd.decide(_pooled(60.0, 30.0, 500, 2.0))["keep"]
    assert not hd.decide(_pooled(120.0, 30.0, 500, 2.0))["keep"]      # p99 at or above 105 mm
    assert not hd.decide(_pooled(60.0, 60.0, 500, 2.0))["keep"]       # overestimate p95 above 52.5 mm
    assert not hd.decide(_pooled(60.0, 30.0, 500, 8.0))["keep"]       # misses 8 % at 105 mm
    # too few frames below 105 mm: judged at 150 mm
    fallback = hd.decide(_pooled(60.0, 30.0, 40, 50.0, close_150=800, missed_150=3.0))
    assert fallback["keep"] and fallback["missed_threshold_mm"] == 150
    nothing = hd.decide(_pooled(60.0, 30.0, 40, 50.0))
    assert not nothing["keep"] and nothing["missed_threshold_mm"] is None
