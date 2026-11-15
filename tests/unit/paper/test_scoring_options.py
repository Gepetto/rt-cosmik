"""score_trial's options for a dataset with its own frame, scored DoF and analysis window."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.config_loader import settings
from rtcosmik.paper import scoring

NAMES = list(settings.joint_angles_names)
KNEE = NAMES.index("Left_Knee_Flexion_Extension[rad]")
CERVICAL = [n for n in NAMES if n.startswith("Cervical")]
FRAMES = 120


def _write(run_dir, q, markers):
    run_dir.mkdir(parents=True)
    with open(run_dir / "joint_angles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(NAMES)
        writer.writerows(q)
    with open(run_dir / "markers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_0"] + [f"{n}_{a}" for n in markers for a in "xyz"])
        for frame in range(len(q)):
            writer.writerow([frame] + [v for n in markers for v in markers[n][frame]])
    return run_dir


@pytest.fixture
def reference(tmp_path):
    t = np.arange(FRAMES) / 60
    q = np.zeros((FRAMES, len(NAMES)))
    q[:, 6] = 1.0
    q[:, KNEE] = 0.6 + 0.5 * np.sin(2 * np.pi * 1.3 * t) + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    rng = np.random.default_rng(0)
    markers = {n: rng.normal(size=3) + np.stack([0.3 * np.sin(3 * t + i), 0.2 * t, np.cos(2 * t)], 1)
               for i, n in enumerate(("RASI", "LKNE", "C7"))}
    return _write(tmp_path / "reference", q, markers), q, markers


def test_cervical_columns_can_be_left_unscored(tmp_path, reference):
    ref_dir, q, markers = reference
    run = q.copy()
    run[:, [NAMES.index(n) for n in CERVICAL]] += np.radians(20.0)
    run[:, KNEE] += np.radians(2.6)
    run_dir = _write(tmp_path / "run", run, markers)

    all_dof, _ = scoring.score_trial(run_dir, ref_dir, 0)
    row, dofs = scoring.score_trial(run_dir, ref_dir, 0, unscored=CERVICAL)

    assert all_dof["scored_dof"] - row["scored_dof"] == 3
    assert not any(d["dof"].startswith("Cervical") for d in dofs)
    assert row["causal_rmse_mean"] == pytest.approx(2.6 / row["scored_dof"], abs=1e-3)


def test_markers_are_compared_in_the_reference_frame(tmp_path, reference):
    from scipy.spatial.transform import Rotation
    ref_dir, q, markers = reference
    R, t = Rotation.from_euler("xyz", [0.4, -1.0, 2.0]).as_matrix(), np.array([3.0, -1.0, 0.5])
    elsewhere = {n: (xyz - t) @ R for n, xyz in markers.items()}      # xyz = R @ elsewhere + t
    run_dir = _write(tmp_path / "run", q, elsewhere)

    apart, _ = scoring.score_trial(run_dir, ref_dir, 0)
    together, _ = scoring.score_trial(run_dir, ref_dir, 0, marker_frame=(R, t))

    assert apart["causal_marker_mm"] > 500
    assert together["causal_marker_mm"] == pytest.approx(0.0, abs=1e-6)


def test_scoring_can_be_restricted_to_a_window_of_frames(tmp_path, reference):
    ref_dir, q, markers = reference
    run = q.copy()
    run[:40, KNEE] += np.radians(9.0)                  # wrong only before the window
    run_dir = _write(tmp_path / "run", run, markers)

    whole, _ = scoring.score_trial(run_dir, ref_dir, 0)
    row, dofs = scoring.score_trial(run_dir, ref_dir, 0, frames=(40, 99))

    assert whole["causal_rmse_mean"] > 0.1
    assert row["causal_rmse_mean"] == pytest.approx(0.0, abs=1e-9)
    assert {d["frames"] for d in dofs} == {60}


def test_lag_search_can_be_bounded(tmp_path, reference):
    ref_dir, q, markers = reference
    late = np.concatenate([np.repeat(q[:1], 9, 0), q[:-9]])
    late_markers = {n: np.concatenate([np.repeat(xyz[:1], 9, 0), xyz[:-9]]) for n, xyz in markers.items()}
    run_dir = _write(tmp_path / "run", late, late_markers)

    free, _ = scoring.score_trial(run_dir, ref_dir, 0, max_lag=30)
    bounded, _ = scoring.score_trial(run_dir, ref_dir, 0, max_lag=4)

    assert (free["lag"], free["marker_lag"]) == (9, 9)
    assert abs(bounded["lag"]) <= 4 and abs(bounded["marker_lag"]) <= 4
