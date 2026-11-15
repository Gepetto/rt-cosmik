"""Checkerboard calibration on rendered views of a board seen by cameras of known pose."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from rtcosmik.calibration import checkerboard as cb  # noqa: E402

SQUARE = 0.1          # m
SQUARES = (7, 8)      # squares along the board's x and y: 6 x 7 inner corners
WIDTH, HEIGHT = 640, 480
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)

# Camera 2 sits 0.8 m to the right of camera 1, turned 20 degrees towards it.
BASELINE = 0.8
TOE_IN = np.radians(20.0)


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def board_texture(pixels_per_square=80):
    """The board with a white margin of one square, and its pixels per metre."""
    nx, ny = SQUARES
    texture = np.full(((ny + 2) * pixels_per_square, (nx + 2) * pixels_per_square), 255, np.uint8)
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                texture[(j + 1) * pixels_per_square:(j + 2) * pixels_per_square,
                        (i + 1) * pixels_per_square:(i + 2) * pixels_per_square] = 0
    return texture, pixels_per_square / SQUARE


def render(board_R, board_T, camera_R=np.eye(3), camera_C=np.zeros(3)):
    """The board, posed in camera 1's frame, seen by a camera at ``camera_C`` with axes ``camera_R``."""
    texture, scale = board_texture()
    R = camera_R.T @ board_R
    T = camera_R.T @ (np.asarray(board_T) - camera_C)
    plane_to_image = K @ np.column_stack([R[:, 0], R[:, 1], T])
    texture_to_plane = np.array([[1 / scale, 0.0, -SQUARE], [0.0, 1 / scale, -SQUARE], [0.0, 0.0, 1.0]])
    image = cv2.warpPerspective(texture, plane_to_image @ texture_to_plane, (WIDTH, HEIGHT),
                                flags=cv2.INTER_AREA, borderValue=255)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def board_poses():
    """Ten board poses in front of both cameras, tilted both ways."""
    tilts = [(0, 0), (25, 0), (-25, 0), (0, 25), (0, -25),
             (20, 20), (-20, 20), (20, -20), (-20, -20), (10, -30)]
    places = [(0.0, -0.4, 2.4), (0.1, -0.3, 2.2), (-0.1, -0.5, 2.6), (0.2, -0.4, 2.8),
              (0.0, -0.2, 2.0), (0.3, -0.5, 2.5), (-0.2, -0.3, 2.3), (0.1, -0.6, 3.0),
              (0.2, -0.2, 2.1), (-0.1, -0.4, 2.7)]
    return [(rot_x(np.radians(a)) @ rot_y(np.radians(b)), np.array(place))
            for (a, b), place in zip(tilts, places)]


@pytest.fixture(scope="module")
def shots(tmp_path_factory):
    """Globs of the two cameras' images of the same ten board poses."""
    root = tmp_path_factory.mktemp("board")
    camera_R, camera_C = rot_y(-TOE_IN), np.array([BASELINE, 0.0, 0.0])
    for name in ("cam_1", "cam_2"):
        (root / name).mkdir()
    for index, (board_R, board_T) in enumerate(board_poses()):
        cv2.imwrite(str(root / "cam_1" / f"img_{index}.png"), render(board_R, board_T))
        cv2.imwrite(str(root / "cam_2" / f"img_{index}.png"),
                    render(board_R, board_T, camera_R, camera_C))
    return str(root / "cam_1" / "*.png"), str(root / "cam_2" / "*.png")


@pytest.fixture(autouse=True)
def board(monkeypatch):
    monkeypatch.setattr(cb.settings, "checkerboard_rows", SQUARES[0] - 1)
    monkeypatch.setattr(cb.settings, "checkerboard_columns", SQUARES[1] - 1)
    monkeypatch.setattr(cb.settings, "checkerboard_square_size", SQUARE)


def test_grid_is_the_inner_corners_one_square_apart():
    size, corners = cb.checkerboard_grid()
    assert size == (6, 7)
    assert corners.shape == (42, 3)
    assert np.allclose(corners[0], [0.0, 0.0, 0.0])
    assert np.allclose(corners[1], [0.1, 0.0, 0.0])
    assert np.allclose(corners[-1], [0.5, 0.6, 0.0])


def test_intrinsics_recover_the_camera(shots):
    reprojection, mtx, dist = cb.calibrate_intrinsics(shots[0], show=False)
    assert reprojection < 0.3
    assert mtx[0, 0] == pytest.approx(600.0, rel=0.02)
    assert mtx[1, 1] == pytest.approx(600.0, rel=0.02)
    assert mtx[0, 2] == pytest.approx(320.0, abs=10.0)
    assert mtx[1, 2] == pytest.approx(240.0, abs=10.0)


def test_stereo_pair_recovers_the_second_camera(shots):
    rmse, R, T = cb.calibrate_stereo_pair(K, NO_DISTORTION, K, NO_DISTORTION, *shots, show=False)
    assert rmse < 0.3
    # p_2 = R p_1 + T: camera 2's centre, in camera 1's frame, is -R^T T.
    assert np.allclose(-R.T @ T.reshape(3), [0.8, 0.0, 0.0], atol=0.01)
    # Its optical axis is turned 20 degrees towards camera 1: sin and cos of 20 degrees.
    assert np.allclose(R.T @ [0.0, 0.0, 1.0], [-0.3420, 0.0, 0.9397], atol=0.01)
    assert np.allclose(R.T @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], atol=0.01)


def test_shared_views_count_only_shots_seen_by_both(shots, tmp_path):
    for name in ("cam_1", "cam_2"):
        (tmp_path / name).mkdir()
    blank = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    for index in range(4):
        for name, pattern in zip(("cam_1", "cam_2"), shots):
            image = cv2.imread(os.path.join(os.path.dirname(pattern), f"img_{index}.png"))
            hidden = (name == "cam_1" and index == 1) or (name == "cam_2" and index == 3)
            cv2.imwrite(str(tmp_path / name / f"img_{index}.png"), blank if hidden else image)

    counted = cb.count_shared_checkerboard_views(
        str(tmp_path / "cam_1" / "*.png"), str(tmp_path / "cam_2" / "*.png"))
    assert counted == (2, 4)


def test_no_board_in_any_image_is_an_error(tmp_path):
    cv2.imwrite(str(tmp_path / "img_0.png"), np.full((HEIGHT, WIDTH, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="no checkerboard"):
        cb.calibrate_intrinsics(str(tmp_path / "*.png"), show=False)


def test_find_checkerboard_without_an_image_is_none():
    assert cb.find_checkerboard(None, (6, 7)) is None


def test_corner_order_must_agree_between_the_two_views():
    forward = np.array([[[0.0, 0.0]], [[10.0, 0.0]], [[10.0, 10.0]]])
    assert cb.corners_consistently_ordered(forward, forward + 50.0)
    assert not cb.corners_consistently_ordered(forward, forward[::-1])
