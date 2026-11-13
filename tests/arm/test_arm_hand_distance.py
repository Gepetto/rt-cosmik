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
    estimate = _dips((100, 110, 0.090), (300, 306, 0.100), (350, 353, 0.100))
    row, _ = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_false_alarm_frames"] == 19
    assert row["causal_105mm_false_alarm_events"] == 2      # the 3-frame flicker is no event
    assert row["causal_105mm_missed_frames"] == 0
    assert row["causal_105mm_approaches"] == 0


def test_a_close_frame_without_an_estimate_is_missed():
    reference = _dips((40, 60, 0.100))
    estimate = _dips((40, 60, 0.100))
    estimate[45:55] = np.nan                                  # dropped frames
    row, series = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_close_frames"] == 20
    assert row["causal_105mm_missed_frames"] == 10
    assert row["causal_105mm_undetected"] == 0
    assert row["missing_frames"] == 10
    assert hd.pool([series])["causal_105mm_missed_pct"] == pytest.approx(50.0)


def test_a_threshold_is_evaluable_on_reference_frames_alone():
    reference = _dips((100, 220, 0.100))                      # 120 frames below 105 mm
    estimate = _dips()
    estimate[100:220] = np.nan
    pooled = hd.pool([hd.score_hand(estimate, reference, _still_effector(), 0, 0)[1]])
    assert pooled["causal_105mm_close_frames"] == 120
    assert pooled["causal_105mm_missed_pct"] == pytest.approx(100.0)
    verdict = hd.decide(pooled)
    assert verdict["missed_threshold_mm"] == 105 and not verdict["missed_ok"]


def test_detection_delay_of_late_approaches():
    reference = _dips((40, 60, 0.100), (120, 150, 0.090), (200, 230, 0.090), (300, 320, 0.1))
    estimate = _dips((45, 60, 0.100), (129, 150, 0.090), (200, 230, 0.090))
    row, series = hd.score_hand(estimate, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_approaches"] == 4
    assert row["causal_105mm_undetected"] == 1
    assert row["causal_105mm_late"] == 2                      # 5 and 9 frames
    assert row["causal_105mm_delay_median_frames"] == pytest.approx(7.0)
    assert row["causal_105mm_delay_p95_frames"] == pytest.approx(8.8)
    pooled = hd.pool([series, series])
    assert pooled["causal_105mm_late"] == 4
    assert pooled["causal_105mm_delay_median_ms"] == pytest.approx(175.0)
    assert pooled["causal_105mm_delay_p95_ms"] == pytest.approx(225.0)


def test_no_late_approach_gives_no_delay():
    reference = _dips((40, 60, 0.100))
    row, _ = hd.score_hand(reference, reference, _still_effector(), 0, 0)
    assert row["causal_105mm_late"] == 0 and np.isnan(row["causal_105mm_delay_median_frames"])


def test_a_missing_reference_frame_or_a_brief_gap_does_not_split_an_approach():
    reference = _dips((40, 60, 0.100), (63, 80, 0.100))       # 3 frames apart
    reference[50] = np.nan
    row, _ = hd.score_hand(_dips(), reference, _still_effector(), 0, 0)
    assert row["causal_105mm_close_frames"] == 36
    assert row["causal_105mm_approaches"] == 1
    assert row["causal_105mm_undetected"] == 1
    assert row["causal_105mm_missed_events"] == 1


def test_a_flicker_shorter_than_the_minimum_is_not_an_approach():
    assert hd.MIN_EVENT_FRAMES == 5
    reference = _dips((100, 104, 0.100), (200, 205, 0.100))
    row, _ = hd.score_hand(_dips(), reference, _still_effector(), 0, 0)
    assert row["causal_105mm_close_frames"] == 9
    assert row["causal_105mm_approaches"] == 1
    assert row["causal_105mm_missed_frames"] == 9
    assert row["causal_105mm_missed_events"] == 1


def test_a_hand_with_no_scored_frame_is_refused():
    estimate = np.full((FRAMES, 3), np.nan)
    with pytest.raises(ValueError, match="no scored frame"):
        hd.score_hand(estimate, _dips(), _still_effector(), 0, 0)


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


# ----------------------------------------------------------------------- script

def _script():
    import importlib.util
    from .conftest import REPO
    path = REPO / "scripts" / "python" / "paper" / "hand_distance.py"
    spec = importlib.util.spec_from_file_location("hand_distance_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    import csv
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_script_records_skipped_trials_and_counts_them_in_the_pool(tmp_path):
    script = _script()
    series = hd.score_hand(_dips(), _dips((40, 60, 0.100)), _still_effector(), 0, 0)
    rows = [{"arm": "arm", "participant": "0001", "task": "RobotWelding", "hand": h, **series[0]}
            for h in script.HANDS]
    distances = {("arm", h, t): [series[1]] for h in script.HANDS for t in ("all", "RobotWelding")}
    skipped = [script.skip("0002", "RobotWelding", "no robot states"),
               script.skip("0003", "RobotPolishing", "no scored frame", "arm", "left")]
    script.write_outputs(tmp_path, ["arm"], rows, distances, skipped)
    assert _read(tmp_path / "skipped.csv") == [
        {"participant": "0002", "task": "RobotWelding", "arm": "all", "hand": "all",
         "reason": "no robot states"},
        {"participant": "0003", "task": "RobotPolishing", "arm": "arm", "hand": "left",
         "reason": "no scored frame"}]
    pooled = {(r["hand"], r["task"]): r for r in _read(tmp_path / "pooled.csv")}
    assert (pooled["left", "all"]["trials"], pooled["left", "all"]["skipped"]) == ("1", "2")
    assert pooled["right", "all"]["skipped"] == "1"
    assert pooled["left", "RobotWelding"]["skipped"] == "1"


def test_script_writes_its_files_when_nothing_was_scored(tmp_path):
    script = _script()
    script.write_outputs(tmp_path, ["arm"], [], {}, [script.skip("0002", "RobotWelding", "x")])
    assert len(_read(tmp_path / "skipped.csv")) == 1
    assert _read(tmp_path / "pooled.csv") == []


def test_script_skips_a_trial_whose_reference_run_is_missing(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script.rd, "robot_frames", lambda *a: {0: np.zeros(7)})
    job = (tmp_path / "runs", tmp_path / "data", "0001", "RobotWelding", ["arm"],
           tmp_path / "frames")
    rows, series, skipped = script.one_trial(job)
    assert (rows, series) == ([], {})
    assert [(s["participant"], s["task"], s["arm"]) for s in skipped] == [
        ("0001", "RobotWelding", "all")]
    assert "reference" in skipped[0]["reason"]
