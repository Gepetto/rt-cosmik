"""World frame from a wand: an ArUco marker on a stick, pointed at known positions."""
import glob

import cv2 as cv
import numpy as np

from rtcosmik.camera.cam_utils import orthonormalize_rotation
from rtcosmik.config_loader import settings


def wand_detector():
    """Detector of the wand's marker (ArUco 4x4)."""
    return cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50),
        cv.aruco.DetectorParameters())


def wand_pose_in_camera(marker_corners, camera_matrix, dist_coeffs):
    """``(tip, rvec, tvec)`` of the wand in camera coordinates from its marker's four corners, or None."""
    half = settings.wand_marker_size / 2
    marker_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv.solvePnP(
        marker_points, np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2),
        camera_matrix, dist_coeffs, False, cv.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rotation, _ = cv.Rodrigues(rvec)
    tip = tvec.reshape(3) + rotation @ np.asarray(settings.wand_tip_offset, dtype=float)
    return tip, rvec, tvec


def wand_positions_in_camera(images_pattern, camera_matrix, dist_coeffs, expected):
    """The wand tip in camera coordinates, one (3,) position per image, in sorted order."""
    names = sorted(glob.glob(images_pattern))
    if len(names) != expected:
        raise ValueError(
            f"need exactly {expected} images to define this frame, found {len(names)}")
    detector = wand_detector()
    positions = []
    for name in names:
        gray = cv.cvtColor(cv.imread(name, 1), cv.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        pose = None if ids is None else \
            wand_pose_in_camera(corners[0], camera_matrix, dist_coeffs)
        if pose is None:
            raise ValueError(f"wand not found in {name}")
        positions.append(pose[0])
    return positions


def frame_from_directions(origin, x_direction, in_plane_direction):
    """Right-handed frame at ``origin``: x along the first direction, z normal to both.

    Returns ``(origin, R)``; the columns of ``R`` are the axes, in the frame the
    directions are given in.
    """
    x = np.asarray(x_direction, dtype=float)
    z = np.cross(x, np.asarray(in_plane_direction, dtype=float))
    y = np.cross(z, x)
    return np.asarray(origin, dtype=float), np.column_stack(
        [axis / np.linalg.norm(axis) for axis in (x, y, z)])


def ground_frame(points):
    """Frame from 3 pointed positions: the origin, a point along x, a point fixing the plane."""
    origin, along_x, in_plane = points
    return frame_from_directions(origin, along_x - origin, in_plane - origin)


def robot_frame(points):
    """Frame from 4 pointed positions: two pairs, each straddling an axis of the robot base.

    The origin is the middle of the first pair; x runs from the middle of the
    second pair towards the origin.
    """
    first_a, first_b, second_a, second_b = points
    origin = (first_a + first_b) / 2
    return frame_from_directions(origin, origin - (second_a + second_b) / 2, first_a - origin)


def invert_pose(R, T):
    """Given ``p_b = R @ p_a + T``, the ``(R, T)`` of ``p_a = R @ p_b + T``."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3)
    return R.T, -R.T @ T


def average_rotations(rotations):
    """Mean of rotation matrices: the rotation nearest to their sum."""
    return orthonormalize_rotation(
        np.sum([np.asarray(R, dtype=float).reshape(3, 3) for R in rotations], axis=0))


def fuse_world_anchor(world_poses, relative_poses, reference,
                      wand_rot_deg=4.0, wand_pos_mm=10.0):
    """Fuse every camera's wand measurement into the reference camera's world pose.

    Rotations are averaged evenly. Positions are weighted by inverse variance,
    a camera's rotation error growing with its distance to the reference:
    ``var = wand_pos**2 + (distance * wand_rot)**2``.

    Args:
        world_poses (dict): ``{camera_id: (R, T)}`` measured, camera in world.
        relative_poses (dict): ``{camera_id: (R, T)}`` with ``p_cam = R @ p_ref + T``.
        reference (int): camera the anchor is for.
        wand_rot_deg, wand_pos_mm (float): expected error of one wand measurement.

    Returns:
        tuple: ``(R, T, spread)``: the reference camera's pose in the world, and
        ``{camera_id: (angle_deg, distance_m)}`` from each camera's own estimate
        of it to the fused one.
    """
    estimates = {}
    baselines = {}
    for camera_id, (Rw, Tw) in world_poses.items():
        if camera_id not in relative_poses:
            continue
        Rw = np.asarray(Rw, dtype=float).reshape(3, 3)
        Tw = np.asarray(Tw, dtype=float).reshape(3)
        R_rel, T_rel = relative_poses[camera_id]
        R_rel = np.asarray(R_rel, dtype=float).reshape(3, 3)
        T_rel = np.asarray(T_rel, dtype=float).reshape(3)
        # p_world = Rw (R_rel p_ref + T_rel) + Tw
        estimates[camera_id] = (Rw @ R_rel, Rw @ T_rel + Tw)
        baselines[camera_id] = float(np.linalg.norm(T_rel))
    if not estimates:
        raise ValueError("no camera has both a world pose and a relative pose")

    R_mean = average_rotations([R for R, _ in estimates.values()])

    sigma_pos = wand_pos_mm / 1000.0
    sigma_rot = np.radians(wand_rot_deg)
    weights = np.array([1.0 / (sigma_pos ** 2 + (baselines[c] * sigma_rot) ** 2)
                        for c in estimates])
    weights /= weights.sum()
    T_mean = np.sum([w * T for w, (_, T) in zip(weights, estimates.values())], axis=0)

    spread = {}
    for camera_id, (R, T) in estimates.items():
        cos = (np.trace(R @ R_mean.T) - 1.0) / 2.0
        spread[camera_id] = (float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                             float(np.linalg.norm(T - T_mean)))
    return R_mean, T_mean, spread
