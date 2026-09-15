"""FootContact with a stand-in network that always predicts contact."""
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.config_loader import settings                  # noqa: E402
from rtcosmik.contact import foot_contact, points            # noqa: E402

WIDTH, HEIGHT = 1280, 720
N_POINTS = len(settings.marker_names) + len(points.JOINT_CANONICAL)


@pytest.fixture
def stub_settings():
    return types.SimpleNamespace(
        N=10, fs=40, order=4, cutoff_freq=10.0, filter_type="lowpass", device="cpu",
        contactvision_path="unused", marker_names=list(settings.marker_names),
        nlf_indices=list(settings.nlf_indices))


@pytest.fixture
def contact(monkeypatch, stub_settings):
    # Always "contact" as far as the network is concerned.
    monkeypatch.setattr(foot_contact, "load_model",
                        lambda path, device: (lambda x: torch.full(x.shape[:2] + (4,), 20.0)))
    return foot_contact.FootContact(stub_settings, num_cameras=2, image_size=(WIDTH, HEIGHT))


def keypoints_in_image():
    """NLF output with every point well inside the image."""
    return np.tile([640.0, 360.0], (N_POINTS, 1)) + np.random.default_rng(0).normal(0, 20, (N_POINTS, 2))


def markers(lift=0.0, shift=0.0):
    """Filtered markers with the feet at `lift` above their standing height."""
    m = np.zeros((len(settings.marker_names), 3))
    m[:, 2] = 1.0
    for name in points.PROBABILITY_MARKERS:
        i = settings.marker_names.index(name)
        m[i] = (shift, 0.1 * i, 0.05 + lift)
    return m


def feed(contact, frames, start=0, **pose):
    """Feed frames with both cameras seeing the feet; return the last output."""
    for k in range(start, start + frames):
        out = contact.update(k / 40.0, [keypoints_in_image()] * 2, markers(**pose))
    return out


def test_output_is_one_row_per_window_node(contact):
    out = contact.update(0.0, [keypoints_in_image()] * 2, markers())
    assert out.shape == (10, len(points.PROBABILITY_MARKERS))


def test_no_contact_until_the_standing_height_is_known(contact):
    for k in range(9):
        out = contact.update(k / 40.0, [keypoints_in_image()] * 2, markers())
        assert np.all(out == 0.0)
    out = feed(contact, 20, start=9)
    assert np.all(out[-1] > 0.99)


def test_a_foot_standing_still_on_the_floor_is_in_contact(contact):
    out = feed(contact, 40)
    assert np.all(out > 0.99)          # the whole window, once filled after calibration


def test_a_raised_foot_is_not(contact):
    feed(contact, 30)
    out = feed(contact, 30, start=30, lift=0.15)
    assert np.all(out < 0.01)


def test_a_fast_foot_is_not(contact):
    feed(contact, 30)
    for k in range(30, 40):            # 1.5 m/s along the floor
        out = contact.update(k / 40.0, [keypoints_in_image()] * 2,
                             markers(shift=1.5 * (k - 29) / 40.0))
    assert np.all(out[-1] < 0.01)


def test_the_window_shifts_with_each_frame(contact):
    """Row k is the probability of the frame the solver holds at node k."""
    feed(contact, 30)
    out = feed(contact, 3, start=30, lift=0.15)
    assert np.all(out[-3:] < 0.01)     # the three lifted frames, newest last
    assert np.all(out[:-4] > 0.99)     # still on the floor before
    np.testing.assert_allclose(contact.latest, out[-1])


def test_no_camera_seeing_the_feet_means_no_contact(contact):
    feed(contact, 30)
    for k in range(30, 60):
        out = contact.update(k / 40.0, [None, None], markers())
    assert np.all(out[-1] == 0.0)


def test_a_camera_that_never_saw_the_feet_does_not_vote(contact, monkeypatch):
    outside = keypoints_in_image()
    outside[:, 0] += 5000.0
    for k in range(40):
        out = contact.update(k / 40.0, [outside, None], markers())
    assert np.all(out == 0.0)


def test_timing_is_bounded_for_four_cameras(monkeypatch, stub_settings):
    """Network excluded."""
    import time
    monkeypatch.setattr(foot_contact, "load_model",
                        lambda path, device: (lambda x: torch.zeros(x.shape[:2] + (4,))))
    contact = foot_contact.FootContact(stub_settings, 4, (WIDTH, HEIGHT))
    kp = [keypoints_in_image()] * 4
    feed_ms = []
    for k in range(200):
        t0 = time.perf_counter()
        contact.update(k / 40.0, kp, markers())
        feed_ms.append((time.perf_counter() - t0) * 1e3)
    assert np.median(feed_ms) < 2.0
