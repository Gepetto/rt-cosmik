"""Upper-arm REBA does not depend on which of the two descriptions of a shoulder posture is stored."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pytest

from rtcosmik.ergonomics import reba_posture as rp


def upper_arm(flexion, abduction):
    """Upper-arm score of a standing person with both shoulders at these angles (deg)."""
    joints = {column: np.zeros(1) for column in rp.ANGLES.values()}
    for side in "rl":
        joints[rp.ANGLES[f"{side}_sh_flex"]] = np.radians([flexion])
        joints[rp.ANGLES[f"{side}_sh_abd"]] = np.radians([abduction])
    points = {"RASI": (0.1, 0.1, 1.0), "LASI": (0.1, -0.1, 1.0), "RPSI": (-0.1, 0.05, 1.0),
              "LPSI": (-0.1, -0.05, 1.0), "C7": (0.0, 0.0, 1.5),
              "RSHO": (0.1, 0.1, 1.4), "LSHO": (0.1, -0.1, 1.4)}
    markers = {f"{name}_{axis}": np.array([value])
               for name, xyz in points.items() for axis, value in zip("xyz", xyz)}
    return int(rp.scores(joints, markers, rp.ABSOLUTE)["upper_arm"][0])


@pytest.mark.parametrize("flexion, abduction, expected", [
    (0, 0, 1), (30, 0, 2), (-30, 0, 2), (60, 0, 3), (100, 0, 4), (60, 20, 4), (100, 20, 5)])
def test_published_upper_arm_levels(flexion, abduction, expected):
    assert upper_arm(flexion, abduction) == expected


@pytest.mark.parametrize("flexion, abduction, expected", [(30, 0, 2), (60, 20, 4), (100, 20, 5)])
def test_other_description_of_the_same_posture_scores_the_same(flexion, abduction, expected):
    assert upper_arm(flexion - 180, 180 - abduction) == expected
    assert upper_arm(flexion + 180, 180 - abduction) == expected


def test_arm_abducted_above_the_shoulder_is_above_shoulder_level():
    assert upper_arm(0, 120) == 5
