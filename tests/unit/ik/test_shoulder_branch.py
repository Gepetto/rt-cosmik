import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.ik import shoulder_branch as sb


@pytest.fixture(scope="module")
def model():
    import example_robot_data as robex
    return robex.human.HumanLoader(height=1.7, weight=70.0, gender="m").robot.model


def _placements(model, q):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    # The two inner shoulder joints differ between branches; every segment does not.
    return np.stack([m.homogeneous for m, name in zip(data.oMi, model.names)
                     if not name.endswith(("shoulder_Z", "shoulder_X"))])


def _pose(model, left, right):
    q = pin.neutral(model)
    for side, triple in (("left", left), ("right", right)):
        q[sb.columns(model, side)[0]] = np.radians(triple)
    return q


def test_other_branch_is_the_same_posture(model):
    q = _pose(model, (40, 120, -30), (-70, 150, 60))
    other = q.copy()
    for side in sb.SIDES:
        cq, _ = sb.columns(model, side)
        other[cq] = sb.other_branch(q[cq])
    assert np.abs(other - q).max() > 1.0
    np.testing.assert_allclose(_placements(model, other), _placements(model, q), atol=1e-9)


def test_nearest_branch_follows_the_reference_across_90_degrees_of_abduction():
    run = np.radians([[40.0, 40.0], [95.0, 95.0], [-30.0, -30.0]])
    reference = np.radians([[41.0, -139.0], [85.0, 86.0], [-29.0, 151.0]])
    matched = np.degrees(sb.nearest_branch(run, reference))
    np.testing.assert_allclose(matched[:, 0], (40, 95, -30), atol=1e-9)
    np.testing.assert_allclose(matched[:, 1], (-140, 85, 150), atol=1e-9)


def _window(model, pinned, side="left", nodes=7):
    q = _pose(model, pinned if side == "left" else (0, 0, 0), pinned if side == "right" else (0, 0, 0))
    X = np.zeros((model.nq + model.nv, nodes))
    X[:model.nq] = q[:, None]
    cq, cv = sb.columns(model, side)
    X[model.nq + np.array(cv), :] = np.array([[0.1], [0.2], [0.3]])
    U = np.zeros((model.nv, nodes))
    U[cv, :] = np.array([[1.0], [2.0], [3.0]])
    return X, U, cq, cv


@pytest.mark.parametrize("side", sb.SIDES)
def test_window_pinned_on_a_limit_moves_to_the_free_branch(model, side):
    X, U, cq, cv = _window(model, (179.5, 150, -45), side)
    before = _placements(model, X[:model.nq, -1].copy())
    reseed = sb.BranchReseed(model)
    assert reseed(X, U) == [side]
    np.testing.assert_allclose(np.degrees(X[cq, -1]), (-0.5, 30, 135), atol=1e-9)
    np.testing.assert_allclose(X[model.nq + np.array(cv), 0], (0.1, -0.2, 0.3))
    np.testing.assert_allclose(U[cv, 0], (1.0, -2.0, 3.0))
    np.testing.assert_allclose(_placements(model, X[:model.nq, -1].copy()), before, atol=1e-9)


def test_window_inside_its_limits_is_left_alone(model):
    X, U, cq, _ = _window(model, (40, 120, -30))
    kept = X.copy()
    assert sb.BranchReseed(model)(X, U) == []
    np.testing.assert_array_equal(X, kept)


def test_no_move_when_the_other_branch_is_outside_the_limits(model):
    X, U, _, _ = _window(model, (179.5, 150, 10))      # other branch: rotation 190 deg
    assert sb.BranchReseed(model)(X, U) == []


def test_a_moved_shoulder_is_held_for_a_while(model):
    X, U, cq, _ = _window(model, (179.5, 150, -45))
    reseed = sb.BranchReseed(model)
    assert reseed(X, U) == ["left"]
    X[cq, :] = np.radians([[179.5], [150], [-45]])
    assert reseed(X, U) == []
    for _ in range(sb.HOLD_FRAMES - 1):
        reseed(X.copy(), U)
    assert reseed(X, U) == ["left"]
