"""Foot contact point layout in NLF's output."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.config_loader import settings          # noqa: E402
from rtcosmik.contact import points                  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_lower_body_indices_follow_the_nlf_output_layout():
    indices = points.lower_body_indices(settings.marker_names, settings.nlf_indices)
    assert len(indices) == len(points.LOWER_BODY) == 13
    n_markers = len(settings.marker_names)
    extra = list(points.JOINT_CANONICAL)
    for name, index in zip(points.LOWER_BODY, indices):
        if name in points.JOINT_CANONICAL:
            # Extra points come after every marker, in JOINT_CANONICAL order.
            assert index == n_markers + extra.index(name)
        else:
            marker, vertex = points.FOOT_MARKERS[name]
            assert settings.marker_names[index] == marker
            assert settings.nlf_indices[index] == vertex


def test_a_foot_marker_on_another_vertex_is_refused():
    nlf_indices = list(settings.nlf_indices)
    nlf_indices[list(settings.marker_names).index("LHEE")] += 1
    with pytest.raises(ValueError, match="LHEE"):
        points.lower_body_indices(settings.marker_names, nlf_indices)


def test_a_missing_foot_marker_is_refused():
    keep = [i for i, n in enumerate(settings.marker_names) if n != "RTOE"]
    names = [settings.marker_names[i] for i in keep]
    nlf_indices = [settings.nlf_indices[i] for i in keep]
    with pytest.raises(ValueError, match="RTOE"):
        points.lower_body_indices(names, nlf_indices)


def test_extra_points_are_seven_canonical_positions():
    extra = points.extra_canonical_points()
    assert extra.shape == (7, 3)
    assert extra.dtype == np.float32


def test_extra_points_are_the_smplx_joints():
    """The stored numbers are the SMPL-X joint regressor applied to NLF's template."""
    regressor = os.path.join(REPO, "weights", "body_models", "smplx", "SMPLX_NEUTRAL.npz")
    if not os.path.isfile(regressor):
        pytest.skip("SMPL-X model files not installed")
    cano = np.load(settings.cano_path)
    joints = np.load(regressor)["J_regressor"][[0, 2, 5, 8, 1, 4, 7]] @ cano
    np.testing.assert_allclose(points.extra_canonical_points(), joints, atol=1e-6)


def test_every_contact_point_is_tracked_and_has_a_probability():
    for frame, source in points.CONTACT_POINTS.items():
        assert frame in settings.keys_to_track_list
        assert source in points.PROBABILITY_MARKERS
