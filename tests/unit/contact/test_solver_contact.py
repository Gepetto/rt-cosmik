"""HumanSolver's side of foot contact that needs no generated solver."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.config_loader import settings            # noqa: E402
from rtcosmik.contact import points                    # noqa: E402
from rtcosmik.pipeline.solver import HumanSolver       # noqa: E402


def configured(**over):
    s = copy.deepcopy(settings)
    for key, value in over.items():
        setattr(s, key, value)
    return s


@pytest.mark.parametrize("over", [
    dict(foot_contact=True, ik_type="mhe", mhe_backend="fatrop"),
    dict(foot_contact=True, ik_type="sbs", mhe_backend="acados"),
])
def test_foot_contact_is_refused_where_it_does_not_exist(over):
    with pytest.raises(ValueError, match="foot_contact needs"):
        HumanSolver(configured(**over))


def test_foot_contact_off_changes_nothing_in_the_solver():
    solver = HumanSolver(configured(foot_contact=False))
    assert solver._contact_frames == []


def test_a_lifted_point_is_held_where_the_foot_is():
    anchors = np.zeros((2, 3))
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    updated = HumanSolver.update_anchors(anchors, positions, [0.0, 0.0], follow=0.05)
    np.testing.assert_allclose(updated, positions)


def test_a_planted_point_follows_only_by_the_follow_rate():
    anchors = np.zeros((1, 3))
    positions = np.array([[1.0, 0.0, 0.0]])
    held = HumanSolver.update_anchors(anchors, positions, [1.0], follow=0.0)
    np.testing.assert_allclose(held, anchors)
    following = HumanSolver.update_anchors(anchors, positions, [1.0], follow=0.05)
    np.testing.assert_allclose(following, [[0.05, 0.0, 0.0]])


def test_partial_contact_blends_hold_and_release():
    anchors = np.zeros((1, 3))
    positions = np.array([[1.0, 0.0, 0.0]])
    updated = HumanSolver.update_anchors(anchors, positions, [0.5], follow=0.1)
    np.testing.assert_allclose(updated, [[0.55, 0.0, 0.0]])   # 1 - 0.5 + 0.5 * 0.1


def test_each_contact_point_reads_its_own_probability():
    solver = HumanSolver(configured(foot_contact=True, ik_type="mhe", mhe_backend="acados"))
    assert solver._contact_frames == list(points.CONTACT_POINTS)
    probability = np.arange(4.0)                     # one value per PROBABILITY_MARKERS entry
    per_point = dict(zip(solver._contact_frames, probability[solver._contact_source]))
    for frame, source in points.CONTACT_POINTS.items():
        assert per_point[frame] == points.PROBABILITY_MARKERS.index(source)
