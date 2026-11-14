import copy
import csv

import numpy as np
import pytest

from .conftest import REFERENCE_RUN, build_trial, needs_comfi

LAG = 8


def _settings(clock_offset=0):
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.ik_type = "sbs"
    s.paper["landmark_map"] = "submitted"      # synthetic trials predate fold maps
    s.order = None                             # no low-pass: any lag is the injected one
    s.paper["clock_offset"] = clock_offset     # built from the reference: no clock offset
    return s


def _replay(trial, repo, clock_offset=0):
    from replay_nlf_arms import replay_trial
    arm = {"name": "nlf", "tag": "nlf_0-2-4-6", "reconstruction": "fuse3d",
           "cameras": trial["cameras"]}
    return replay_trial(trial["dataset"], trial["views"], trial["out_dir"], trial["participant"],
                        trial["task"], arm, _settings(clock_offset), repo=repo)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    import subprocess
    repo = tmp_path_factory.mktemp("repo")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    return repo


@pytest.fixture(scope="module")
def rows(tmp_path_factory, repo):
    out = {}
    for lag in (0, LAG):
        trial = build_trial(tmp_path_factory.mktemp(f"lag{lag}"), frames=400, lag=lag,
                            source=REFERENCE_RUN)
        out[lag] = (_replay(trial, repo), trial["out_dir"])
    return out


@needs_comfi
def test_injected_lag_counts_in_causal_error_and_alignment_removes_it(rows):
    (on_time, _), (late, _) = rows[0], rows[LAG]
    assert (on_time["marker_lag"], late["marker_lag"]) == (0, LAG)
    assert late["lag"] - on_time["lag"] == LAG
    assert late["causal_rmse_mean"] > on_time["causal_rmse_mean"] + 0.5
    assert abs(late["aligned_rmse_mean"] - on_time["aligned_rmse_mean"]) < 0.3


@needs_comfi
def test_bias_and_sd_decompose_the_rmse(rows):
    _, run_dir = rows[LAG]
    dofs = list(csv.DictReader(open(run_dir / "per_dof.csv")))
    assert len(dofs) == 29
    for kind in ("causal", "aligned"):
        rmse, bias, sd = (np.array([float(d[f"{kind}_{k}_deg"]) for d in dofs])
                          for k in ("rmse", "bias", "sd"))
        np.testing.assert_allclose(rmse ** 2, bias ** 2 + sd ** 2, rtol=1e-9)


@needs_comfi
def test_row_reports_markers_jitter_failures_and_limits(rows):
    row, _ = rows[0]
    for key in ("causal_marker_mm", "aligned_marker_mm", "jitter_deg", "solver_failures",
                "violation_pct", "dropped_frames"):
        assert row[key] != "" and np.isfinite(float(row[key])), key
    assert row["solver_failures"] == 0


@needs_comfi
def test_clock_offset_is_measured_on_the_recording_and_not_charged_to_the_arm(tmp_path, repo):
    from measure_clock_offsets import measure
    trial = build_trial(tmp_path, frames=400, lag=LAG, source=REFERENCE_RUN)

    offset = measure(trial["views"], trial["dataset"], trial["participant"], trial["task"])
    row = _replay(trial, repo, clock_offset=offset)

    assert offset == LAG
    assert row["marker_lag"] == 0


def _write_run(path, names, q):
    path.mkdir(parents=True)
    with open(path / "joint_angles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(q.tolist())
    with open(path / "markers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_0", "C7_x", "C7_y", "C7_z"])
        writer.writerows([[i, 0.0, 0.0, 1.0] for i in range(len(q))])


def _score_left_shoulder(tmp_path, run_deg, reference_deg):
    from rtcosmik.config_loader import settings
    from rtcosmik.paper.scoring import score_trial
    names = list(settings.joint_angles_names)
    t = np.linspace(0, 6, 200)
    q = np.zeros((len(t), len(names)))
    q[:, 6] = 1.0
    for i, name in enumerate(names):
        if name.endswith("[rad]"):
            q[:, i] = 0.3 * np.sin(t + i)
    shoulder = [i for i, name in enumerate(names) if name.startswith("Left_Shoulder")]
    run, reference = q.copy(), q.copy()
    run[:, shoulder], reference[:, shoulder] = np.radians(run_deg), np.radians(reference_deg)
    _write_run(tmp_path / "ref", names, reference)
    _write_run(tmp_path / "run", names, run)
    _, dofs = score_trial(tmp_path / "run", tmp_path / "ref", 0)
    return {d["dof"]: d["causal_rmse_deg"] for d in dofs}


def test_a_shoulder_in_the_other_branch_scores_as_the_same_posture(tmp_path):
    errors = _score_left_shoulder(tmp_path, (40, 120, -30), (-140, 60, 150))
    assert max(errors.values()) < 1e-6


def test_shoulders_either_side_of_90_degrees_of_abduction_score_their_true_difference(tmp_path):
    errors = _score_left_shoulder(tmp_path, (40, 95, -30), (-140, 95, 150))
    assert errors["Left_Shoulder_Abduction_Adduction"] == pytest.approx(10.0, abs=1e-6)
    assert errors["Left_Shoulder_Flexion_Extension"] < 1e-6


def test_reference_runs_can_live_outside_the_dataset(tmp_path, monkeypatch):
    from rtcosmik.paper.scoring import reference_dir
    assert reference_dir("/data", "1012", "Lifting").parts[:2] == ("/", "data")
    monkeypatch.setenv("RTCOSMIK_REFERENCE_DIR", str(tmp_path))
    assert reference_dir("/data", "1012", "Lifting") == tmp_path / "1012" / "Lifting" / "mocap_reference"
