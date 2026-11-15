"""Errors between camera extrinsics, and their perturbation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.paper.calibration_study import perturbed, pose_errors, rotation_angle


def turn_z(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_errors_of_a_turned_and_displaced_camera():
    # Reference: a camera 2 m along x, axes aligned. Estimate: turned 3 deg, 2.1 m away
    # in a direction 90 deg off.
    reference = (np.eye(3), np.array([-2.0, 0.0, 0.0]))
    R = turn_z(3.0)
    estimate = (R, -R @ np.array([0.0, 2.1, 0.0]))
    errors = pose_errors(estimate, reference)
    assert errors["rotation_deg"] == pytest.approx(3.0)
    assert errors["baseline_m"] == pytest.approx(2.0)
    assert errors["length_pct"] == pytest.approx(5.0)
    assert errors["direction_deg"] == pytest.approx(90.0)
    assert errors["position_mm"] == pytest.approx(2900.0)


def test_identical_poses_have_no_error():
    pose = (turn_z(40.0), np.array([0.3, -1.0, 2.0]))
    errors = pose_errors(pose, pose)
    assert errors["rotation_deg"] == pytest.approx(0.0, abs=1e-5)
    assert errors["position_mm"] == pytest.approx(0.0, abs=1e-9)


def test_perturbation_has_the_requested_size():
    R, T = turn_z(25.0), np.array([1.0, 2.0, 3.0])
    new_R, new_T = perturbed(R, T, 2.0, 0.05, np.random.default_rng(3))
    assert rotation_angle(new_R @ R.T) == pytest.approx(2.0)
    assert np.linalg.norm(new_T - T) == pytest.approx(0.05)
    assert np.allclose(new_R @ new_R.T, np.eye(3))
