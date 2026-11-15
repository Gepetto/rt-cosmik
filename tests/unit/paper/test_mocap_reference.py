"""The reference's head markers, from the mocap head band."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rtcosmik.folds import FOLDS
from rtcosmik.paper import mocap_reference as mr

# A level band 0.2 m wide centred at (1, 2, 1.6) m, tilted 30 deg forward about the lateral axis.
TILT = np.radians(30.0)
NORMAL = np.array([np.sin(TILT), 0.0, np.cos(TILT)])
FORWARD = np.array([np.cos(TILT), 0.0, -np.sin(TILT)])
CENTRE = np.array([1.0, 2.0, 1.6])
BAND = {"FHD": CENTRE + 0.1 * FORWARD, "BHD": CENTRE - 0.1 * FORWARD,
        "LHD": CENTRE + [0.0, 0.1, 0.0], "RHD": CENTRE - [0.0, 0.1, 0.0]}


def _trial(tmp_path):
    header = ["Frame"] + [f"{name}_{axis}[mm]" for name in BAND for axis in "XYZ"]
    row = ["0"] + [str(1e3 * v) for name in BAND for v in BAND[name]]
    (tmp_path / "markers_trajectories.csv").write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    return tmp_path


def test_head_is_the_band_centre_raised_along_the_band_normal(tmp_path):
    _, names, data = mr.load_mocap_markers(_trial(tmp_path), ["Head", "REar", "LEar"], head_raise=0.08)

    np.testing.assert_allclose(data[0, names.index("Head")], CENTRE + 0.08 * NORMAL, atol=1e-9)
    np.testing.assert_allclose(data[0, names.index("REar")], BAND["RHD"], atol=1e-9)


def test_head_stays_at_the_band_centre_without_a_raise(tmp_path):
    _, names, data = mr.load_mocap_markers(_trial(tmp_path), ["Head"])

    np.testing.assert_allclose(data[0, names.index("Head")], CENTRE, atol=1e-9)


def test_a_participant_takes_the_raise_of_the_fold_that_holds_them():
    for fold, participants in enumerate(FOLDS):
        assert {mr.head_raise(p) for p in participants} == {mr.HEAD_RAISE_M[fold]}
    assert all(0.07 < value < 0.09 for value in mr.HEAD_RAISE_M)
