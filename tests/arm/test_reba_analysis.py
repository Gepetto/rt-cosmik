"""Posture-only REBA analyses on synthetic trials with REBA levels known by construction.

Expected scores are read off the published tables (Hignett & McAtamney, 2000):

STAND  everything neutral: A(neck 1, trunk 1, legs 1) = 1, B(upper 1, lower 2, wrist 1) = 1, C = 1
MID    trunk 40, knees 40, neck 30, shoulders 60: A(2, 3, 2) = 5, B(3, 2, 1) = 4, C = 5
DEEP   trunk 70, knees 70, neck 30, shoulders 100: A(2, 4, 3) = 7, B(4, 2, 1) = 5, C = 9
"""
import csv
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from . import conftest  # noqa: F401  (puts src on the path)

STAND = dict(trunk=0, knee=0, neck=0, shoulder=0)
MID = dict(trunk=40, knee=40, neck=30, shoulder=60)
DEEP = dict(trunk=70, knee=70, neck=30, shoulder=100)
NEGLIGIBLE, LOW, MEDIUM, HIGH = 0, 1, 2, 3


def timeline(*segments):
    """Per-frame postures from ``(posture, frames)`` segments."""
    return [posture for posture, frames in segments for _ in range(frames)]


def delayed(postures, lag):
    return postures[:1] * lag + postures[:len(postures) - lag]


def write_run(folder, postures, trunk_offset=0.0, shoulder_offset=0.0, no_angles=(),
              no_shoulder=()):
    """A run folder (joint_angles.csv, markers.csv) holding the postures, one per frame.

    Frames in ``no_angles`` have no joint angles, those in ``no_shoulder`` no RSHO marker.
    """
    from rtcosmik.ergonomics.reba_posture import ANGLES
    folder.mkdir(parents=True)
    value = {"neck_flex": "neck", "r_knee": "knee", "l_knee": "knee",
             "r_sh_flex": "shoulder", "l_sh_flex": "shoulder"}
    offset = {"r_sh_flex": shoulder_offset, "l_sh_flex": shoulder_offset}
    with open(folder / "joint_angles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ANGLES.values())
        for frame, p in enumerate(postures):
            writer.writerow([np.nan if frame in no_angles
                             else np.radians(p[value[k]] + offset.get(k, 0)) if k in value else 0.0
                             for k in ANGLES])
    pelvis = {"RASI": (0.1, 0.1, 1.0), "LASI": (0.1, -0.1, 1.0),
              "RPSI": (-0.1, 0.05, 1.0), "LPSI": (-0.1, -0.05, 1.0)}
    names = [*pelvis, "C7", "RSHO", "LSHO"]
    with open(folder / "markers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_0"] + [f"{n}_{a}" for n in names for a in "xyz"])
        for frame, p in enumerate(postures):
            t = np.radians(p["trunk"] + trunk_offset)
            c7 = (0.5 * np.sin(t), 0.0, 1.0 + 0.5 * np.cos(t))
            sho = [(x, y, z + 0.4) for x, y, z in (pelvis["RASI"], pelvis["LASI"])]
            if frame in no_shoulder:
                sho[0] = (np.nan,) * 3
            writer.writerow([frame, *np.ravel([*pelvis.values(), c7, *sho])])
    return folder


def analyse(tmp_path, estimate, reference, clock_offset=0, **offsets):
    from rtcosmik.paper import reba_analysis as ra
    pairs, lag = ra.paired_scores(write_run(tmp_path / "estimate", estimate, **offsets),
                                  write_run(tmp_path / "reference", reference), clock_offset)
    return ra, pairs, lag


REFERENCE = timeline((STAND, 100), (DEEP, 100), (STAND, 100))
UNDERSTATED = timeline((STAND, 100), (DEEP, 25), (MID, 75), (STAND, 100))


def test_postures_score_what_the_reba_tables_say(tmp_path):
    _, pairs, _ = analyse(tmp_path, UNDERSTATED, REFERENCE)
    for angles in ("absolute", "neutral_reference", "neutral_own"):
        pair = pairs["causal", angles]
        assert [pair["ref"]["reba"][i] for i in (50, 110, 150)] == [1, 9, 9]
        assert [pair["est"]["reba"][i] for i in (50, 110, 150)] == [1, 9, 5]
        assert [pair["est"]["risk"][i] for i in (50, 110, 150)] == [NEGLIGIBLE, HIGH, MEDIUM]


def test_confusion_matrix_counts_frames_by_reference_then_estimate_level(tmp_path):
    ra, pairs, _ = analyse(tmp_path, UNDERSTATED, REFERENCE)
    pair = pairs["causal", "absolute"]
    result = ra.agreement(pair["est"]["reba"], pair["ref"]["reba"])
    expected = np.zeros((5, 5), int)
    expected[NEGLIGIBLE, NEGLIGIBLE], expected[HIGH, HIGH], expected[HIGH, MEDIUM] = 200, 25, 75
    assert np.array_equal(result["confusion"], expected)
    assert np.allclose(ra.by_reference_level(result["confusion"])[HIGH], [0, 0, 0.75, 0.25, 0])
    assert np.isnan(ra.by_reference_level(result["confusion"])[LOW]).all()
    assert result["risk_agree_pct"] == pytest.approx(75.0)
    assert result["reba_bias"] == pytest.approx(-1.0)
    assert result["reba_mae"] == pytest.approx(1.0)
    # observed weighted disagreement 0.0625, expected by chance 0.3125
    assert result["risk_kappa"] == pytest.approx(0.8)


def test_high_risk_underestimation_is_the_share_rated_lower_and_the_signed_error(tmp_path):
    ra, pairs, _ = analyse(tmp_path, UNDERSTATED, REFERENCE)
    pair = pairs["causal", "neutral_own"]
    result = ra.agreement(pair["est"]["reba"], pair["ref"]["reba"])
    assert result["high_frames"] == 100
    assert result["high_under_pct"] == pytest.approx(75.0)
    assert result["high_missed_pct"] == pytest.approx(75.0)
    assert result["high_bias"] == pytest.approx(-3.0)       # 75 frames at 5 - 9


def test_no_high_risk_frames_gives_no_underestimation_figure(tmp_path):
    ra, pairs, _ = analyse(tmp_path, timeline((STAND, 150), (MID, 150)),
                           timeline((STAND, 150), (MID, 150)))
    pair = pairs["causal", "absolute"]
    result = ra.agreement(pair["est"]["reba"], pair["ref"]["reba"])
    assert result["high_frames"] == 0 and np.isnan(result["high_under_pct"])


def test_lag_counts_in_the_causal_comparison_and_alignment_removes_it(tmp_path):
    ra, pairs, lag = analyse(tmp_path, delayed(REFERENCE, 8), REFERENCE)
    causal = ra.agreement(*(pairs["causal", "absolute"][k]["reba"] for k in ("est", "ref")))
    aligned = ra.agreement(*(pairs["aligned", "absolute"][k]["reba"] for k in ("est", "ref")))
    assert lag == 8
    assert causal["risk_agree_pct"] == pytest.approx(100 * (300 - 16) / 300)
    assert causal["high_under_pct"] == pytest.approx(8.0)   # late into the posture
    assert aligned["risk_agree_pct"] == 100.0 and aligned["high_under_pct"] == 0.0
    assert aligned["frames"] == 292


def test_clock_offset_is_removed_before_the_causal_comparison(tmp_path):
    ra, pairs, lag = analyse(tmp_path, delayed(REFERENCE, 3), REFERENCE, clock_offset=3)
    causal = ra.agreement(*(pairs["causal", "absolute"][k]["reba"] for k in ("est", "ref")))
    assert lag == 0
    assert causal["risk_agree_pct"] == 100.0 and causal["frames"] == 297


def test_constant_offsets_count_with_absolute_angles_and_cancel_with_own_neutral(tmp_path):
    # +10 deg trunk and +30 deg shoulder standing: A(1, 2, 1) = 2, B(2, 2, 1) = 2, C = 2 (low)
    ra, pairs, _ = analyse(tmp_path, REFERENCE, REFERENCE, trunk_offset=10, shoulder_offset=30)
    score = {angles: ra.agreement(*(pairs["causal", angles][k]["reba"] for k in ("est", "ref")))
             for angles in ("absolute", "neutral_reference", "neutral_own")}
    for angles in ("absolute", "neutral_reference"):
        assert score[angles]["confusion"][NEGLIGIBLE, LOW] == 200
        assert score[angles]["reba_bias"] == pytest.approx(200 / 300)
    assert score["neutral_own"]["risk_agree_pct"] == 100.0
    assert score["neutral_own"]["reba_mae"] == 0.0


def test_representative_postures_are_chosen_on_the_estimate_not_the_reference(tmp_path):
    estimate = timeline((STAND, 100), (MID, 100), (DEEP, 40), (STAND, 60))
    reference = timeline((STAND, 100), (DEEP, 100), (MID, 40), (STAND, 60))
    ra, pairs, _ = analyse(tmp_path, estimate, reference)
    rows = {r["selection"]: r for r in ra.representative_postures(pairs["causal", "absolute"],
                                                                  hold_frames=20)}
    peak = rows["peak"]
    assert 200 <= peak["frame"] < 240                       # where the estimate peaks
    assert (peak["est_reba"], peak["ref_reba"]) == (9, 5)
    assert (peak["est_risk"], peak["ref_risk"]) == (HIGH, MEDIUM)
    medium = rows["level_medium"]
    assert 100 <= medium["frame"] < 200
    assert (medium["est_reba"], medium["ref_reba"]) == (5, 9)
    assert medium["est_share_pct"] == pytest.approx(100 * 100 / 300)
    assert rows["level_negligible"]["typical"] and not medium["typical"]
    assert set(rows) == {"peak", "level_negligible", "level_medium", "level_high"}


def test_peak_posture_must_be_held(tmp_path):
    estimate = timeline((STAND, 100), (MID, 20), (DEEP, 2), (MID, 78), (STAND, 100))
    ra, pairs, _ = analyse(tmp_path, estimate, estimate)
    peak = ra.representative_postures(pairs["causal", "absolute"], hold_frames=20)[0]
    assert peak["selection"] == "peak" and peak["est_reba"] == 5
    assert 100 <= peak["frame"] < 200


def test_peak_is_the_held_score_when_a_brief_spike_sits_mid_spell(tmp_path):
    estimate = timeline((STAND, 100), (MID, 49), (DEEP, 2), (MID, 49), (STAND, 100))
    ra, pairs, _ = analyse(tmp_path, estimate, estimate)
    peak = ra.representative_postures(pairs["causal", "absolute"], hold_frames=20)[0]
    assert peak["selection"] == "peak" and peak["held_reba"] == 5
    assert 100 <= peak["frame"] < 200
    assert (peak["est_reba"], peak["est_risk"]) == (5, MEDIUM)   # a frame at the held score


def test_a_trial_holding_no_score_has_no_peak(tmp_path):
    estimate = timeline((STAND, 30), (MID, 30))
    gaps = range(10, 60, 15)                                 # no 20 scored frames in a row
    ra, pairs, _ = analyse(tmp_path, estimate, estimate, no_shoulder=gaps)
    rows = ra.representative_postures(pairs["causal", "neutral_own"], hold_frames=20)
    assert {r["selection"] for r in rows} == {"level_negligible", "level_medium"}


def test_absolute_angles_leave_frames_with_a_missing_marker_unscored(tmp_path):
    ra, pairs, _ = analyse(tmp_path, REFERENCE, REFERENCE, no_shoulder=range(120, 130))
    scored = {angles: np.isfinite(pairs["causal", angles]["est"]["reba"])
              for angles in ("absolute", "neutral_own")}
    assert not scored["absolute"][120:130].any()
    assert np.array_equal(scored["absolute"], scored["neutral_own"])
    assert scored["absolute"].sum() == 290


def test_absolute_scoring_is_asked_for_by_name(tmp_path):
    from rtcosmik.ergonomics import reba_posture as rp
    joints, points = ra_tables(write_run(tmp_path / "run", REFERENCE))
    with pytest.raises(TypeError):
        rp.scores(joints, points)
    with pytest.raises(ValueError, match="neutral"):
        rp.scores(joints, points, None)
    assert rp.scores(joints, points, rp.ABSOLUTE)["reba"][150] == 9


def ra_tables(run_dir):
    from rtcosmik.paper import reba_analysis as ra
    return ra._tables(run_dir)[2:]


def test_a_trial_without_a_neutral_posture_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="neutral posture"):
        analyse(tmp_path, REFERENCE, REFERENCE, no_angles=range(20))


def test_a_trial_with_no_frame_scored_by_both_is_refused(tmp_path):
    from rtcosmik.paper import reba_analysis as ra
    estimate = write_run(tmp_path / "estimate", REFERENCE, no_angles=range(10, 300))
    reference = write_run(tmp_path / "reference", REFERENCE, no_angles=range(10))
    with pytest.raises(ValueError, match="no frame scored"):
        ra.paired_scores(estimate, reference, 0)


SETTINGS = SimpleNamespace(paper={"reba_hold_s": 0.5, "clock_offset": 0}, fs=40)


def run_script(tmp_path, estimates):
    """The analysis script on ``{participant: write_run arguments}``, reference REFERENCE."""
    from rtcosmik.paper import reba_analysis as ra
    path = conftest.REPO / "scripts" / "python" / "paper" / "reba_analyses.py"
    spec = importlib.util.spec_from_file_location("reba_analyses", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for participant, kwargs in estimates.items():
        write_run(tmp_path / "runs" / participant / "Lifting" / "arm", **kwargs)
        write_run(tmp_path / "data" / "output" / "campaign" / participant / "Lifting"
                  / "mocap_reference", REFERENCE)
    out = tmp_path / "out"
    done, failed = script.analyse(ra, SETTINGS, tmp_path / "data", "arm",
                                  script.trials_of(tmp_path / "runs", "arm"), out)
    return done, failed, out


def test_script_lists_failed_trials_and_trials_without_a_held_peak(tmp_path):
    flicker = timeline((STAND, 100), *[(DEEP, 5), (STAND, 5)] * 10, (STAND, 100))
    done, failed, out = run_script(tmp_path, {
        "0001": dict(postures=REFERENCE),
        "0002": dict(postures=REFERENCE, no_angles=range(20)),
        "0003": dict(postures=flicker, no_shoulder=range(0, 300, 15))})
    assert done == 2 and len(failed) == 1
    report = (out / "summary.md").read_text()
    assert "Failed trials" in report and "0002/Lifting" in report and "neutral posture" in report
    assert "No held peak" in report and "0003/Lifting" in report
    assert "0001/Lifting" not in report
    assert "| frames |" in report


def test_script_survives_every_trial_failing(tmp_path):
    done, failed, out = run_script(tmp_path, {"0002": dict(postures=REFERENCE,
                                                           no_angles=range(20))})
    assert done == 0 and "neutral posture" in failed[0]


def test_summary_pools_frames_per_participant_then_averages_participants():
    from rtcosmik.paper import reba_analysis as ra
    trials = [("0001", [9, 9, 5, 5], [9, 9, 9, 9]), ("0001", [1, 1], [1, 1]),
              ("0002", [9, 9], [9, 9])]
    participants, across, pooled = ra.summarize(trials)
    assert participants["0001"]["high_under_pct"] == pytest.approx(50.0)
    assert participants["0001"]["frames"] == 6
    mean, sd, n = across["high_under_pct"]
    assert (mean, n) == (pytest.approx(25.0), 2) and sd == pytest.approx(50 / np.sqrt(2))
    assert pooled["high_under_pct"] == pytest.approx(100 * 2 / 6)
