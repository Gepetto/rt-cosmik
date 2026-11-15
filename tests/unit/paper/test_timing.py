"""Timing of a paced replay and of recorded camera timestamps."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rtcosmik.paper import timing


def test_stage_statistics_give_median_percentiles_and_maximum():
    stats = timing.statistics(np.arange(1.0, 102.0))          # 1 .. 101 ms

    assert stats == {"median": 51.0, "p95": 96.0, "p99": 100.0, "max": 101.0}


def test_a_frame_processed_within_the_period_is_ready_after_its_own_processing_time():
    latency = timing.paced_latency([10.0, 20.0, 5.0], period_ms=25.0)

    np.testing.assert_allclose(latency, [10.0, 20.0, 5.0])


def test_a_slow_frame_delays_the_frames_behind_it_until_the_backlog_clears():
    # frame 0 takes 60 ms: frame 1 (arrives at 25) starts at 60, frame 2 (arrives at 50) at 70
    latency = timing.paced_latency([60.0, 10.0, 10.0, 10.0], period_ms=25.0)

    np.testing.assert_allclose(latency, [60.0, 45.0, 30.0, 15.0])
    assert timing.missed_deadlines(latency, period_ms=25.0) == 3


def test_a_stream_at_the_nominal_rate_misses_no_frame():
    times = np.arange(200)[:, None] * 0.025 + np.array([0.0, 0.004])

    report = timing.timestamp_report(times, period_s=0.025)

    assert report["missing_frames"] == [0, 0]
    assert report["long_intervals"] == [0, 0]
    assert report["spread_ms"]["median"] == pytest.approx(4.0)
    assert report["offset_to_first_ms"] == pytest.approx([0.0, 4.0])


def test_a_gap_of_three_periods_misses_two_frames_on_that_camera():
    times = np.arange(100)[:, None] * 0.025 + np.zeros(2)
    times[40:, 1] += 0.050

    report = timing.timestamp_report(times, period_s=0.025)

    assert report["missing_frames"] == [0, 2]
    assert report["long_intervals"] == [0, 1]


def test_a_late_frame_followed_by_an_early_one_is_a_long_interval_but_no_missing_frame():
    times = (np.arange(101) * 0.025)[:, None].copy()
    times[50] += 0.015                                        # intervals of 40 then 10 ms

    report = timing.timestamp_report(times, period_s=0.025)

    assert report["missing_frames"] == [0]
    assert report["long_intervals"] == [1]
    assert report["interval_ms"][0]["mean"] == pytest.approx(25.0)
    assert report["interval_ms"][0]["max"] == pytest.approx(40.0)
