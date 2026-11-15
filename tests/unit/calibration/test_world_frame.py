"""World frame from the wand: frame construction, fusion, and the tip from a rendered marker."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from rtcosmik.calibration import world_frame as wf  # noqa: E402

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_invert_pose_maps_points_back():
    R, T = rot_z(np.pi / 2), np.array([1.0, 0.0, 0.0])
    R_inv, T_inv = wf.invert_pose(R, T)
    # (1, 2, 3) goes to (-1, 1, 3) under a quarter turn about z then +1 in x.
    assert np.allclose(R_inv @ np.array([-1.0, 1.0, 3.0]) + T_inv, [1.0, 2.0, 3.0])
    assert np.allclose(T_inv, [0.0, 1.0, 0.0])


def test_frame_axes_follow_the_directions():
    origin, R = wf.frame_from_directions([1.0, 2.0, 3.0], [0.0, 2.0, 0.0], [-5.0, 1.0, 0.0])
    assert np.allclose(origin, [1.0, 2.0, 3.0])
    # x along +y, the plane's other direction towards -x, so z = x cross y is +z.
    assert np.allclose(R, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_frame_ignores_the_length_of_the_directions():
    _, short = wf.frame_from_directions(np.zeros(3), [2.0, 0.0, 0.0], [0.0, 5.0, 0.0])
    _, long = wf.frame_from_directions(np.zeros(3), [9.0, 0.0, 0.0], [3.0, 0.1, 0.0])
    assert np.allclose(short, np.eye(3)) and np.allclose(long, np.eye(3))


def test_ground_frame_x_runs_from_the_origin_to_the_second_point():
    points = [np.array([1.0, 1.0, 0.0]), np.array([3.0, 1.0, 0.0]), np.array([2.0, 4.0, 0.0])]
    origin, R = wf.ground_frame(points)
    assert np.allclose(origin, [1.0, 1.0, 0.0])
    assert np.allclose(R, np.eye(3))


def test_robot_frame_x_runs_from_the_second_pair_towards_the_origin():
    points = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
              np.array([-2.0, 1.0, 0.0]), np.array([-2.0, -1.0, 0.0])]
    origin, R = wf.robot_frame(points)
    assert np.allclose(origin, [0.0, 0.0, 0.0])   # middle of the first pair
    assert np.allclose(R, np.eye(3))              # x away from the second pair, y to the first point


def test_average_of_rotations_spread_evenly_is_the_middle_one():
    mean = wf.average_rotations([rot_z(0.3), rot_z(0.4), rot_z(0.5)])
    assert np.allclose(mean, rot_z(0.4), atol=1e-9)


def test_average_of_one_rotation_is_itself():
    assert np.allclose(wf.average_rotations([rot_y(1.1)]), rot_y(1.1))


def two_cameras():
    """Two cameras with known world poses, and camera 2 relative to camera 0."""
    R0, T0 = rot_z(0.6), np.array([0.5, -1.0, 2.0])
    R2, T2 = rot_y(1.3) @ rot_z(-0.2), np.array([-1.0, 0.4, 2.2])
    # p_2 = R2^T (p_world - T2) with p_world = R0 p_0 + T0
    relative = {0: (np.eye(3), np.zeros(3)), 2: (R2.T @ R0, R2.T @ (T0 - T2))}
    return {0: (R0, T0), 2: (R2, T2)}, relative


def test_exact_measurements_fuse_to_the_reference_pose():
    measured, relative = two_cameras()
    R, T, spread = wf.fuse_world_anchor(measured, relative, reference=0)
    assert np.allclose(R, measured[0][0], atol=1e-9)
    assert np.allclose(T, measured[0][1], atol=1e-9)
    assert max(angle for angle, _ in spread.values()) < 1e-6
    assert max(distance for _, distance in spread.values()) < 1e-9


def test_positions_are_weighted_by_inverse_variance():
    # Camera 2 is 1 m away; with 10 mm wand error and a rotation error worth
    # sqrt(3) * 10 mm at 1 m, its variance is 4 times the reference's: weights 0.8, 0.2.
    measured = {0: (np.eye(3), np.zeros(3)), 2: (np.eye(3), np.array([-1.0, 0.5, 0.0]))}
    relative = {0: (np.eye(3), np.zeros(3)), 2: (np.eye(3), np.array([1.0, 0.0, 0.0]))}
    _, T, spread = wf.fuse_world_anchor(
        measured, relative, reference=0,
        wand_rot_deg=np.degrees(0.01 * np.sqrt(3)), wand_pos_mm=10.0)
    assert np.allclose(T, [0.0, 0.1, 0.0])
    assert spread[0][1] == pytest.approx(0.1)
    assert spread[2][1] == pytest.approx(0.4)


def test_a_distant_camera_rotation_error_stays_out_of_the_position():
    measured, relative = two_cameras()
    R2, T2 = measured[2]
    noisy = {0: measured[0], 2: (rot_y(np.radians(6)) @ R2, T2)}
    _, T, spread = wf.fuse_world_anchor(noisy, relative, reference=0)
    # Camera 2's own estimate is off by 6 degrees over its ~2 m baseline.
    assert spread[2][1] > 0.1
    assert np.linalg.norm(T - measured[0][1]) < 0.01


def test_cameras_without_a_relative_pose_are_left_out():
    measured, relative = two_cameras()
    _, _, spread = wf.fuse_world_anchor(measured, {0: relative[0]}, reference=0)
    assert set(spread) == {0}


def test_no_common_camera_is_an_error():
    measured, _ = two_cameras()
    with pytest.raises(ValueError):
        wf.fuse_world_anchor(measured, {}, reference=0)


# --- the wand tip, from rendered images of its marker ------------------------

MARKER_SIZE = 0.2
TIP_OFFSET = (0.0, 0.3, 0.0)


@pytest.fixture
def wand(monkeypatch):
    monkeypatch.setattr(wf.settings, "wand_marker_size", MARKER_SIZE)
    monkeypatch.setattr(wf.settings, "wand_tip_offset", TIP_OFFSET)


def render_marker(centre, right, down):
    """A 640x480 view of the marker: its centre and its image-right and image-down edges (m, camera frame)."""
    centre, right, down = (np.asarray(v, dtype=float) for v in (centre, right, down))
    half = MARKER_SIZE / 2
    corners = [centre - half * right - half * down, centre + half * right - half * down,
               centre + half * right + half * down, centre - half * right + half * down]
    pixels = np.array([(K @ p)[:2] / p[2] for p in corners], dtype=np.float32)

    side, margin = 400, 100
    texture = np.full((side + 2 * margin,) * 2, 255, np.uint8)
    texture[margin:-margin, margin:-margin] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side)
    source = np.array([[margin, margin], [margin + side, margin],
                       [margin + side, margin + side], [margin, margin + side]], np.float32)
    image = cv2.warpPerspective(texture, cv2.getPerspectiveTransform(source, pixels),
                                (640, 480), flags=cv2.INTER_AREA, borderValue=255)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_wand_tip_is_one_offset_along_the_marker_up_edge(wand, tmp_path):
    # Upright and facing the camera, 1 m away: the tip is 0.3 m above the centre (camera y is down).
    cv2.imwrite(str(tmp_path / "img_0.png"), render_marker([0.1, 0.1, 1.0], [1, 0, 0], [0, 1, 0]))
    # Turned a quarter turn in the image: the marker's up edge points to the camera's right.
    cv2.imwrite(str(tmp_path / "img_1.png"), render_marker([-0.2, 0.0, 1.2], [0, 1, 0], [-1, 0, 0]))

    tips = wf.wand_positions_in_camera(str(tmp_path / "*.png"), K, NO_DISTORTION, expected=2)

    assert np.allclose(tips[0], [0.1, -0.2, 1.0], atol=0.02)
    assert np.allclose(tips[1], [0.1, 0.0, 1.2], atol=0.02)


def test_wrong_number_of_images_is_an_error(wand, tmp_path):
    cv2.imwrite(str(tmp_path / "img_0.png"), render_marker([0, 0, 1.0], [1, 0, 0], [0, 1, 0]))
    with pytest.raises(ValueError, match="exactly 3"):
        wf.wand_positions_in_camera(str(tmp_path / "*.png"), K, NO_DISTORTION, expected=3)


def test_image_without_the_wand_is_an_error(wand, tmp_path):
    cv2.imwrite(str(tmp_path / "img_0.png"), np.full((480, 640, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="wand not found"):
        wf.wand_positions_in_camera(str(tmp_path / "*.png"), K, NO_DISTORTION, expected=1)
