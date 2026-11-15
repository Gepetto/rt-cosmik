"""Camera extrinsics from a person walking in the scene.

Each camera's metric 3D landmarks are registered rigidly onto the reference
camera's, over the whole recording.
"""
from dataclasses import dataclass

import numpy as np

#: Least standard deviation of the body centre along its main direction of travel, in metres.
MIN_TRAVEL = 0.5
#: Fewest frames in which a camera and the reference must both see the person.
MIN_SHARED_FRAMES = 30
#: Landmarks further than this many median residuals from the fit are left out of it.
OUTLIER_FACTOR = 3.0
FOOT_LANDMARKS = ("RHEE", "LHEE", "RTOE", "LTOE", "R5MHD", "L5MHD")


class DegenerateRecording(ValueError):
    """The recording cannot support a calibration."""


@dataclass
class HumanCalibration:
    """``poses[c]`` is ``(R, T)`` with ``p_c = R @ p_reference + T``; ``points`` the
    views averaged in the reference camera's frame, (F, J, 3) in metres;
    ``residual_m[c]`` the median distance left between camera c and the reference."""
    poses: list
    points: np.ndarray
    travel_m: float
    residual_m: list


def travel(poses3d):
    """Standard deviation of the body centre along its main direction, in metres, for one camera."""
    points = np.asarray(poses3d, float)
    centres = points[np.isfinite(points).all(axis=(1, 2))].mean(axis=1)
    if len(centres) < 2:
        return 0.0
    spread = np.linalg.svd(centres - centres.mean(axis=0), compute_uv=False)
    return float(spread[0] / np.sqrt(len(centres)))


def rigid_fit(source, target):
    """Least-squares ``(R, T)`` with ``target = R @ source + T``, for (N, 3) points."""
    centre_s, centre_t = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((target - centre_t).T @ (source - centre_s))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, centre_t - R @ centre_s


def register_pair(poses3d_a, poses3d_b):
    """Pose of camera b relative to camera a, ``p_b = R @ p_a + T``, and the median residual (m)."""
    shared = np.isfinite(poses3d_a).all(axis=(1, 2)) & np.isfinite(poses3d_b).all(axis=(1, 2))
    if shared.sum() < MIN_SHARED_FRAMES:
        raise DegenerateRecording(f"only {int(shared.sum())} frames show the person to both "
                                  f"cameras (need {MIN_SHARED_FRAMES})")
    a, b = poses3d_a[shared].reshape(-1, 3), poses3d_b[shared].reshape(-1, 3)
    keep = np.ones(len(a), bool)
    for _ in range(4):
        R, T = rigid_fit(a[keep], b[keep])
        residual = np.linalg.norm(a @ R.T + T - b, axis=1)
        keep = residual <= OUTLIER_FACTOR * max(np.median(residual), 1e-9)
    return R, T, float(np.median(residual))


def calibrate(poses3d, min_travel=MIN_TRAVEL):
    """Extrinsics of every camera relative to the first, from one recording.

    Args:
        poses3d: (F, C, J, 3) metric landmarks in each camera's own frame, NaN
            where a camera saw no one.
        min_travel: least standard deviation of the body centre accepted, in metres.

    Raises:
        DegenerateRecording: the person stayed in place, or too few frames are shared.
    """
    poses3d = np.asarray(poses3d, float)
    distance = travel(poses3d[:, 0])
    if distance < min_travel:
        raise DegenerateRecording(
            f"the person's position varies by {distance:.2f} m only (standard deviation along "
            f"the walk; need {min_travel} m): record them walking across the area the "
            f"cameras cover")
    poses, residuals = [(np.eye(3), np.zeros(3))], [0.0]
    for camera in range(1, poses3d.shape[1]):
        R, T, residual = register_pair(poses3d[:, 0], poses3d[:, camera])
        poses.append((R, T))
        residuals.append(residual)
    in_reference = np.stack([(poses3d[:, c] - T) @ R for c, (R, T) in enumerate(poses)])
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(in_reference).sum(axis=0)
        points = np.where(counts > 0, np.nansum(in_reference, axis=0) / np.maximum(counts, 1),
                          np.nan)
    return HumanCalibration(poses, points, distance, residuals)


def floor_frame(points, marker_names):
    """World pose of the reference camera, ``p_world = R @ p_reference + T``.

    z is up from the floor, x the camera's optical axis laid on the floor, and
    the origin is the point of the floor under the camera.
    """
    points, names = np.asarray(points, float), list(marker_names)
    missing = [n for n in ("Head",) + FOOT_LANDMARKS if n not in names]
    if missing:
        raise ValueError(f"landmarks needed for the floor are missing: {missing}")
    feet = points[:, [names.index(n) for n in FOOT_LANDMARKS]].reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(axis=1)]
    if len(feet) < 8:
        raise DegenerateRecording("not enough foot landmarks to fit a floor")
    centre = np.median(feet, axis=0)
    for _ in range(3):      # leave out the feet in the air
        z = np.linalg.svd(feet - centre)[2][-1]
        height = np.abs((feet - centre) @ z)
        feet = feet[height <= np.percentile(height, 80)]
        centre = feet.mean(axis=0)
    head = points[:, names.index("Head")]
    if np.nanmedian((head - centre) @ z) < 0:
        z = -z
    x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), np.array([0.0, 0.0, -float(centre @ z)])
