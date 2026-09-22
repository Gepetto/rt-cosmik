"""Recording control, now that it lives with the data rather than the viewer."""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.saver.recorder import Recorder      # noqa: E402


def make_settings(directory, **over):
    base = dict(SAVE_CSV=True, SAVE_VID=False, SAVE_DIR=directory,
                record_hotkeys=False, record_on_start=False,
                marker_names=["RASI", "LASI"],
                joint_angles_names=["j0", "j1", "j2"])
    base.update(over)
    return SimpleNamespace(**base)


def frame():
    return {"RASI": np.array([1.0, 2.0, 3.0]), "LASI": np.array([4.0, 5.0, 6.0])}


def test_records_nothing_until_enabled():
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d), num_cameras=2) as rec:
            assert rec.record([0, 0], frame(), [0.1, 0.2, 0.3]) is False
            assert rec.rows == 0


def test_toggling_starts_and_stops_recording():
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d), num_cameras=2) as rec:
            rec.set_enabled(True)
            assert rec.record([1, 1], frame(), [0.1, 0.2, 0.3]) is True
            rec.set_enabled(False)
            assert rec.record([2, 2], frame(), [0.1, 0.2, 0.3]) is False
            assert rec.rows == 1


def test_record_on_start_needs_no_toggle():
    """What a headless or scripted run wants: no keyboard involved."""
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d, record_on_start=True), num_cameras=2) as rec:
            assert rec.enabled
            assert rec.record([3, 3], frame(), [0.1, 0.2, 0.3]) is True


def test_shared_flag_follows_the_toggle():
    """The video writers are separate processes reading this flag."""
    flag = SimpleNamespace(value=False)
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d), num_cameras=1, saving_flag=flag) as rec:
            assert flag.value is False
            rec.set_enabled(True)
            assert flag.value is True
            rec.set_enabled(False)
            assert flag.value is False


def test_csv_carries_a_frame_counter_per_camera():
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d, record_on_start=True), num_cameras=3) as rec:
            rec.record([7, 8, 9], frame(), [0.1, 0.2, 0.3])
        header = open(os.path.join(d, "markers.csv")).readline().strip()
        assert header.startswith("Frame_0,Frame_1,Frame_2,")
        row = open(os.path.join(d, "markers.csv")).readlines()[1].strip()
        assert row.startswith("7,8,9,")


def test_no_csv_when_saving_is_off():
    with tempfile.TemporaryDirectory() as d:
        with Recorder(make_settings(d, SAVE_CSV=False, record_on_start=True),
                      num_cameras=1) as rec:
            assert rec.record([0], frame(), [0.1, 0.2, 0.3]) is False
        assert not os.path.exists(os.path.join(d, "markers.csv"))
