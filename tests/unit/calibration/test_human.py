"""Human-pose calibration on synthetic walks seen by cameras of known extrinsics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.calibration.human import (FOOT_LANDMARKS, DegenerateRecording, calibrate,
                                        floor_frame)

NAMES = ["Head"] + list(FOOT_LANDMARKS) + [f"M{i}" for i in range(12)]


def look_at(position, target):
    """World-to-camera (R, T) of a camera at ``position`` looking at ``target``, z up."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ position


def body():
    """A 1.75 m body: head, six foot landmarks on the floor, twelve points in between."""
    rng = np.random.default_rng(0)
    feet = np.array([[-0.05, -0.1, 0], [-0.05, 0.1, 0], [0.15, -0.1, 0],
                     [0.15, 0.1, 0], [0.1, -0.15, 0], [0.1, 0.15, 0]], float)
    middle = rng.uniform([-0.15, -0.25, 0.3], [0.15, 0.25, 1.5], (12, 3))
    return np.vstack([[0.0, 0.0, 1.75], feet, middle])


def walk(path):
    """(F, J, 3) world points of the body moved along ``path`` (F, 2) on the floor."""
    return body()[None] + np.concatenate([path, np.zeros((len(path), 1))], axis=1)[:, None]


def circuit(frames=240):
    angle = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    return np.stack([1.2 * np.cos(angle), 1.0 * np.sin(angle)], axis=1)


CAMERAS = [look_at(np.array(p, float), np.array([0.0, 0.0, 1.0]))
           for p in ([3.5, -0.4, 1.4], [3.5, 0.4, 1.4], [-3.5, 0.4, 1.4], [-3.5, -0.4, 1.4])]


def observe(points, noise_m=0.0, seed=1):
    """Camera-frame poses (F, C, J, 3) of world points."""
    rng = np.random.default_rng(seed)
    poses = [points @ R.T + T + rng.normal(0, noise_m, points.shape) for R, T in CAMERAS]
    return np.stack(poses, axis=1)


def relative(camera):
    """Expected (R, T) of ``camera`` with respect to camera 0: p_cam = R p_0 + T."""
    R0, T0 = CAMERAS[0]
    R, T = CAMERAS[camera]
    return R @ R0.T, T - R @ R0.T @ T0


def errors(poses):
    """Rotation errors (deg) and translation errors (m) of cameras 1..3."""
    rotation, translation = [], []
    for camera in range(1, len(CAMERAS)):
        R, T = relative(camera)
        cosine = (np.trace(poses[camera][0] @ R.T) - 1) / 2
        rotation.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        translation.append(np.linalg.norm(poses[camera][1] - T))
    return np.array(rotation), np.array(translation)


STRAIGHT = np.stack([np.linspace(-1.5, 1.5, 240), np.zeros(240)], axis=1)


@pytest.mark.parametrize("path", [circuit(), STRAIGHT])
def test_exact_observations_recover_the_extrinsics(path):
    result = calibrate(observe(walk(path)))
    rotation, translation = errors(result.poses)
    assert np.allclose(result.poses[0][0], np.eye(3)) and np.allclose(result.poses[0][1], 0)
    assert rotation.max() < 1e-4
    assert translation.max() < 1e-5


def test_noisy_observations_stay_within_tolerance():
    result = calibrate(observe(walk(circuit()), noise_m=0.05))
    rotation, translation = errors(result.poses)
    assert rotation.max() < 0.5
    assert translation.max() < 0.02


def test_a_view_that_misplaces_the_person_in_some_frames_is_tolerated():
    poses3d = observe(walk(circuit()))
    poses3d[::10, 2] += [0.0, 0.0, 1.0]
    rotation, translation = errors(calibrate(poses3d).poses)
    assert rotation.max() < 0.05
    assert translation.max() < 0.005


def test_a_person_who_stays_in_place_is_rejected():
    with pytest.raises(DegenerateRecording):
        calibrate(observe(walk(np.zeros((240, 2)))))


def test_the_travel_required_can_be_lowered():
    result = calibrate(observe(walk(circuit() * 0.3)), min_travel=0.2)
    assert errors(result.poses)[0].max() < 1e-4


def test_too_few_shared_frames_are_rejected():
    poses3d = observe(walk(circuit()))
    poses3d[5:, 2] = np.nan
    with pytest.raises(DegenerateRecording):
        calibrate(poses3d)


def test_floor_frame_puts_the_feet_at_zero_and_the_head_up():
    result = calibrate(observe(walk(circuit())))
    R, T = floor_frame(result.points, NAMES)
    world = result.points @ R.T + T
    feet = world[:, [NAMES.index(n) for n in FOOT_LANDMARKS], 2]
    assert np.abs(feet).max() < 0.005
    assert np.allclose(world[:, NAMES.index("Head"), 2], 1.75, atol=0.005)
    # The reference camera stands at its height above the floor.
    assert T[2] == pytest.approx(1.4, abs=0.005)
