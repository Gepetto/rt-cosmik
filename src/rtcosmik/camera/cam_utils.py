"""Camera discovery and calibration loading.

Calibration is stored in the COMFI layout, which is the canonical format for
RT-COSMIK both online (``config/cam_params``) and offline (one participant
directory of a recorded dataset)::

    <root>/intrinsics/camera_<i>_intrinsics.yaml
    <root>/extrinsics/cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml

Intrinsics are OpenCV ``FileStorage`` documents; cam-to-world files are plain
YAML. Poses are taken from ``cam_to_world`` rather than by chaining the
``cam_to_cam`` files the layout may also contain: one step covers any subset of
cameras, and cam-to-cam chains are not always complete.
"""

import logging
import os
import subprocess

import cv2 as cv
import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)

# Cameras are identified by their even v4l2 index in the COMFI layout.
DEFAULT_CAMERA_IDS = (0, 2, 4, 6)


def list_cameras():
    """
    Use v4l2-ctl to list all connected cameras and their device paths.
    Returns a dictionary of camera indices and associated device names.
    """
    cameras = {}
    try:
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        devices = output.split("\n\n")
        for device in devices:
            lines = device.split("\n")
            if len(lines) > 1:
                device_name = lines[0].strip()
                video_path = lines[1].strip()
                if "/dev/video" in video_path:
                    index = int(video_path.split("video")[-1])
                    cap = cv.VideoCapture(index, cv.CAP_V4L2)
                    if cap.isOpened():
                        cameras[index] = device_name
                    cap.release()
    except Exception as e:
        print("Error using v4l2-ctl:", e)
    return cameras


def orthonormalize_rotation(R):
    """Project a matrix onto the nearest rotation matrix (SVD, det = +1).

    Cam-to-world files store rotations as 6-decimal text, so the parsed matrix
    is only orthonormal to ~1e-6 (det can be off by ~2e-7). That is far below
    calibration accuracy, but it leaves the reference camera's pose relative to
    itself slightly off identity, so relative poses are derived from cleaned
    rotations rather than the raw parsed values.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:  # guard against a reflection
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


# ---------------------------------------------------------------------------
# Single-file readers
# ---------------------------------------------------------------------------

def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Args:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - dist_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()

    if camera_matrix is None or dist_matrix is None:
        raise ValueError(f"Missing 'K'/'D' in intrinsics file: {path}")

    return camera_matrix, dist_matrix


def load_cam_pose(filename):
    """
    Load a camera pose from a cam-to-world YAML file.

    The file holds the transform that takes a point expressed in the camera
    frame into the world frame::

        camera_extrinsics:
          frame_from: camera_0
          frame_to: world
          rotation_matrix: [[...], [...], [...]]
          translation_vector: [tx, ty, tz]

    so that ``p_world = R @ p_cam + T``. The pose is used as stored; it is not
    inverted here.

    Args:
        filename (str): The path to the YAML file.
    Returns:
        tuple:
            - rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            - translation_vector (np.ndarray): The translation vector, shape (3,).
    """
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Empty cam-to-world file: {filename}")

    extrinsics = data.get('camera_extrinsics', data)

    try:
        rotation_matrix = np.array(extrinsics['rotation_matrix'], dtype=float).reshape((3, 3))
        translation_vector = np.array(extrinsics['translation_vector'], dtype=float).reshape((3,))
    except KeyError as exc:
        raise ValueError(
            f"Missing 'rotation_matrix'/'translation_vector' in cam-to-world file: {filename}"
        ) from exc

    return rotation_matrix, translation_vector


def load_soder_transform(filename):
    """
    Load a camera pose from the ``soder.txt`` written by the Procrustes fit.

    This is the source the aggregated ``camera_<i>_extrinsics.yaml`` is
    generated from (its ``source_file`` field names it), and it carries the same
    camera-to-world convention::

        # Transformation Parameters
        Rotation Matrix (R):
        r00 r01 r02
        ...
        Translation Vector (d):
        tx ty tz
        Scale Factor (s): 1.000000
        RMS Error: 0.001438

    Args:
        filename (str): The path to the soder.txt file.
    Returns:
        tuple:
            - rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            - translation_vector (np.ndarray): The translation vector, shape (3,).
    """
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    rows = []
    translation = None
    for index, line in enumerate(lines):
        if line.startswith('Rotation Matrix'):
            rows = [[float(v) for v in lines[index + offset].split()] for offset in (1, 2, 3)]
        elif line.startswith('Translation Vector'):
            translation = [float(v) for v in lines[index + 1].split()]

    if len(rows) != 3 or translation is None:
        raise ValueError(f"Could not parse rotation/translation from soder file: {filename}")

    return np.array(rows, dtype=float).reshape((3, 3)), np.array(translation, dtype=float).reshape((3,))


# ---------------------------------------------------------------------------
# Layout-aware loaders
# ---------------------------------------------------------------------------

def intrinsics_path(config_path, camera_id):
    """Path to one camera's intrinsics file."""
    return os.path.join(config_path, "intrinsics", f"camera_{camera_id}_intrinsics.yaml")


def cam_to_world_path(config_path, camera_id, calib_session=None):
    """
    Resolve one camera's cam-to-world pose file.

    Prefers the aggregated ``camera_<i>_extrinsics.yaml``. Some recordings never
    had it generated and instead keep the raw Procrustes output, either directly
    in the camera directory or split across ``calib_<n>`` subdirectories when the
    session was calibrated more than once.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        camera_id (int): Camera to resolve.
        calib_session (str | None): Name of the ``calib_<n>`` subdirectory to use
            when several exist. ``None`` takes the first in sorted order. The same
            session is used for every camera so the world frame stays consistent.

    Returns:
        str | None: Path to a readable pose file, or None if there is none.
    """
    camera_dir = os.path.join(
        config_path, "extrinsics", "cam_to_world", f"camera_{camera_id}"
    )

    aggregated = os.path.join(camera_dir, f"camera_{camera_id}_extrinsics.yaml")
    if os.path.isfile(aggregated):
        return aggregated

    direct_soder = os.path.join(camera_dir, "soder.txt")
    if os.path.isfile(direct_soder):
        return direct_soder

    if not os.path.isdir(camera_dir):
        return None

    sessions = sorted(
        entry for entry in os.listdir(camera_dir)
        if os.path.isfile(os.path.join(camera_dir, entry, "soder.txt"))
    )
    if not sessions:
        return None

    if calib_session is not None:
        if calib_session not in sessions:
            raise FileNotFoundError(
                f"Calibration session {calib_session!r} not found for camera {camera_id} "
                f"in {camera_dir} (available: {', '.join(sessions)})"
            )
        chosen = calib_session
    else:
        chosen = sessions[0]
        if len(sessions) > 1:
            LOGGER.warning(
                "Camera %d in %s has %d calibration sessions (%s); using %s. "
                "Pass calib_session to choose explicitly.",
                camera_id, camera_dir, len(sessions), ", ".join(sessions), chosen,
            )

    return os.path.join(camera_dir, chosen, "soder.txt")


def load_cam_to_world(filename):
    """Load a cam-to-world pose from either an extrinsics YAML or a soder.txt."""
    if os.path.basename(filename) == "soder.txt":
        return load_soder_transform(filename)
    return load_cam_pose(filename)


def load_camera_parameters(config_path, camera_ids=DEFAULT_CAMERA_IDS, calib_session=None):
    """
    Load intrinsics and extrinsics for a set of cameras.

    Every camera pose is read from ``cam_to_world`` and then re-expressed
    relative to the first requested camera, which becomes the reference frame
    that triangulation outputs points in.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        camera_ids (Sequence[int]): Cameras to load, in order. The first one is
            the reference frame.
        calib_session (str | None): Calibration session to use when a camera has
            several, see :func:`cam_to_world_path`.

    Returns:
        tuple: ``(mtxs, dists, projections, rotations, translations)``, each a
        list ordered like ``camera_ids``. ``projections[i]`` is the 3x4 matrix
        ``[R | T]`` *without* the intrinsics: triangulation undistorts to
        normalized coordinates, so K must not be baked in.
    """
    camera_ids = list(camera_ids)

    if not camera_ids:
        raise ValueError(f"No cameras requested or found under {config_path}")

    mtxs = []
    dists = []
    world_poses = []
    for camera_id in camera_ids:
        intrinsics_file = intrinsics_path(config_path, camera_id)
        if not os.path.isfile(intrinsics_file):
            raise FileNotFoundError(f"Missing intrinsics for camera {camera_id}: {intrinsics_file}")
        pose_file = cam_to_world_path(config_path, camera_id, calib_session)
        if pose_file is None:
            raise FileNotFoundError(
                f"No cam-to-world pose for camera {camera_id} under {config_path}"
            )

        K, D = load_cam_params(intrinsics_file)
        mtxs.append(np.asarray(K, dtype=float))
        dists.append(np.asarray(D, dtype=float))
        R_world, T_world = load_cam_to_world(pose_file)
        world_poses.append((orthonormalize_rotation(R_world), T_world))

    # Reference camera: p_world = R_ref @ p_ref + T_ref
    R_ref, T_ref = world_poses[0]

    rotations = []
    translations = []
    projections = []
    for R_cam, T_cam in world_poses:
        # p_cam = R_cam^T (p_world - T_cam), and p_world = R_ref p_ref + T_ref,
        # so p_cam = (R_cam^T R_ref) p_ref + R_cam^T (T_ref - T_cam).
        rotation = R_cam.T @ R_ref
        translation = (R_cam.T @ (T_ref - T_cam)).reshape(3, 1)

        rotations.append(rotation)
        translations.append(translation)
        projections.append(np.concatenate([rotation, translation], axis=-1))

    return mtxs, dists, projections, rotations, translations


def load_world_transformation(config_path, ref_camera=DEFAULT_CAMERA_IDS[0], calib_session=None):
    """
    Load the transform from the reference camera frame to the world frame.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        ref_camera (int): Camera whose frame triangulation outputs points in,
            i.e. the first entry of the ``camera_ids`` passed to
            :func:`load_camera_parameters`.
        calib_session (str | None): Calibration session to use when the camera
            has several, see :func:`cam_to_world_path`.

    Returns:
        tuple: ``(world_R_cam, world_T_cam)`` with shapes (3, 3) and (3,), such
        that ``p_world = world_R_cam @ p_cam + world_T_cam``.
    """
    pose_file = cam_to_world_path(config_path, ref_camera, calib_session)
    if pose_file is None:
        raise FileNotFoundError(
            f"No cam-to-world pose for reference camera {ref_camera} under {config_path}"
        )
    world_R_cam, world_T_cam = load_cam_to_world(pose_file)
    return orthonormalize_rotation(world_R_cam), world_T_cam.reshape((3,))
