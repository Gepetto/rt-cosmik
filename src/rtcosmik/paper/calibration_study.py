"""Comparing camera extrinsics, and perturbing them."""
import numpy as np


def rotation_angle(R):
    """Angle of a rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pose_errors(estimated, reference):
    """Errors of one camera's pose relative to the reference camera.

    Both are ``(R, T)`` with ``p_camera = R @ p_reference + T``. Returns the
    rotation error (deg), the error on the camera's position (mm), the baseline
    (m), the error on its length (%) and on its direction (deg).
    """
    (R, T), (R_ref, T_ref) = estimated, reference
    T, T_ref = np.asarray(T, float).reshape(3), np.asarray(T_ref, float).reshape(3)
    centre, centre_ref = -R.T @ T, -R_ref.T @ T_ref
    baseline = float(np.linalg.norm(centre_ref))
    cosine = centre @ centre_ref / (np.linalg.norm(centre) * baseline)
    return {"rotation_deg": rotation_angle(R @ R_ref.T),
            "position_mm": 1e3 * float(np.linalg.norm(centre - centre_ref)),
            "baseline_m": baseline,
            "length_pct": 100 * (float(np.linalg.norm(centre)) / baseline - 1),
            "direction_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))}


def _unit(rng):
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def perturbed(R, T, angle_deg, shift_m, rng):
    """A camera's world pose turned by ``angle_deg`` and moved by ``shift_m``, in random directions."""
    import cv2 as cv
    turn = cv.Rodrigues(_unit(rng) * np.radians(angle_deg))[0]
    return turn @ np.asarray(R, float), np.asarray(T, float).reshape(3) + shift_m * _unit(rng)
