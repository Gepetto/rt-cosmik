"""Camera discovery and calibration loading.

**Camera pose convention.** RT-COSMIK expects the pose of the camera *in the
world frame*: ``R`` is the camera's orientation in world coordinates (its columns
are the camera axes expressed in the world frame) and ``T`` is the camera's
position in world coordinates, in metres. Equivalently the pair maps a point from
camera coordinates into world coordinates::

    p_world = R @ p_cam + T

This is the opposite of what ``cv2.solvePnP`` and most aruco helpers return, so
their output must be inverted before being stored (``R = R_cv.T``,
``T = -R_cv.T @ t_cv``). Getting it backwards raises no error; the subject is
simply reconstructed in the wrong place.

Calibration is stored in the COMFI layout, canonical for RT-COSMIK both online
(``config/cam_params``) and offline (one participant directory of a dataset)::

    <root>/intrinsics/camera_<i>_intrinsics.yaml
    <root>/extrinsics/cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
    <root>/extrinsics/cam_to_cam/camera_<a>_to_camera_<b>.yaml

Intrinsics are OpenCV ``FileStorage`` documents; cam-to-world files are plain
YAML; cam-to-cam files are ``cv2.stereoCalibrate`` output stored in OpenCV's own
convention (``p_b = R @ p_a + T``), i.e. saved exactly as OpenCV writes them.

Reconstruction only needs the cameras' poses *relative to each other*; the world
frame enters once, through :func:`load_world_transformation`. Those relative
poses come from either source:

* a world pose per camera, as a fit against shared motion-capture markers gives.
  Each camera is placed independently, so error does not accumulate. Preferred.
* stereo pairs chained from the reference camera, as a checkerboard calibration
  gives. Then only the *reference* camera needs a world pose -- typically from a
  single aruco marker -- to place the whole rig.
"""
import logging
import os
from collections import deque
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


# ---------------------------------------------------------------------------
# Which physical camera is which
# ---------------------------------------------------------------------------
#
# The ids in the layout are v4l2 indices, which record the order the kernel
# happened to enumerate devices in. They are not identity: replugging a camera,
# or simply rebooting, can hand index 0 to a different camera. Nothing in the
# calibration would notice, so one camera's intrinsics and pose would be applied
# to another, producing a plausible-looking but wrong reconstruction.
#
# A calibration may therefore carry a ``cameras.yaml`` recording the hardware
# behind each id when it was calibrated. When present it is used to map the
# indices seen now onto the ids they were calibrated as.


def camera_manifest_path(config_path):
    """Path to the record of which physical camera each id refers to."""
    return os.path.join(config_path, "cameras.yaml")


def load_camera_manifest(config_path):
    """Read a calibration's camera manifest, or None when it has none."""
    path = camera_manifest_path(config_path)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    entries = data.get("cameras", [])
    return {int(entry["id"]): entry for entry in entries if "id" in entry}


def camera_bus_info(index):
    """The USB port path behind a v4l2 index, or None.

    The port path is what distinguishes otherwise identical cameras: units of
    the same model commonly share a placeholder serial number, so a serial
    cannot tell two of them apart.
    """
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "--info"],
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None
    for line in output.splitlines():
        if "Bus info" in line:
            return line.split(":", 1)[1].strip()
    return None


def resolve_camera_ids(config_path, indices):
    """Map the v4l2 indices present now onto the ids they were calibrated as.

    Recabling a rig changes which index each camera answers to, but not where
    each camera sits in the room, so the calibration remains valid and is merely
    attached to the wrong ids. Matching on the recorded port path recovers the
    pairing, turning a reshuffle into a remap instead of a silent error.

    This recovers *cable* changes only. A camera physically moved to a new place
    has stale extrinsics whatever its port says, and only recalibration fixes
    that; no amount of USB metadata can detect it.

    Args:
        config_path (str): calibration root, possibly holding ``cameras.yaml``.
        indices (Sequence[int]): v4l2 indices present now.

    Returns:
        dict: ``{index: calibrated_camera_id}``. Without a manifest, or for an
        index whose port was never calibrated, the index maps to itself, which
        is the behaviour of a calibration that predates the manifest.
    """
    manifest = load_camera_manifest(config_path)
    if manifest is None:
        return {index: index for index in indices}

    by_bus = {entry["bus_info"]: camera_id
              for camera_id, entry in manifest.items() if entry.get("bus_info")}
    resolved = {}
    for index in indices:
        bus = camera_bus_info(index)
        camera_id = by_bus.get(bus) if bus else None
        if camera_id is None:
            LOGGER.warning(
                "Camera at index %d is on port %s, which is not in the calibration's "
                "manifest. Using its index as its id; if the rig was recabled this "
                "pairs it with the wrong calibration.", index, bus or "unknown")
            resolved[index] = index
        else:
            if camera_id != index:
                LOGGER.warning(
                    "Camera at index %d is the one calibrated as camera_%d (port %s); "
                    "using camera_%d's calibration for it.",
                    index, camera_id, bus, camera_id)
            resolved[index] = camera_id
    return resolved


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


def cam_to_cam_path(config_path, cam_a, cam_b):
    """Path to the stereo result relating two cameras, or None if absent."""
    candidate = os.path.join(
        config_path, "extrinsics", "cam_to_cam", f"camera_{cam_a}_to_camera_{cam_b}.yaml")
    return candidate if os.path.isfile(candidate) else None


def load_cam_to_cam(filename):
    """
    Load a stereo (camera-to-camera) result written by ``cv2.stereoCalibrate``.

    The file is an OpenCV ``FileStorage`` document holding both cameras'
    intrinsics plus the pose relating them::

        p_b = R @ p_a + T

    i.e. it takes a point expressed in camera A's frame into camera B's frame.

    Returns:
        tuple: ``(R, T)`` with shapes (3, 3) and (3,).
    """
    storage = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    rotation = storage.getNode("R").mat()
    translation = storage.getNode("T").mat()
    storage.release()
    if rotation is None or translation is None:
        raise ValueError(f"Missing 'R'/'T' in cam-to-cam file: {filename}")
    return np.asarray(rotation, dtype=float).reshape(3, 3), \
        np.asarray(translation, dtype=float).reshape(3)


def chain_relative_poses(config_path, camera_ids):
    """
    Derive each camera's pose relative to the first, by chaining stereo results.

    A checkerboard calibration produces a pose per *pair* of cameras, not a world
    pose per camera, so the poses form a graph that has to be walked. Links are
    usable in both directions -- ``camera_0_to_camera_2`` inverted relates 2 to 0
    -- so this walks breadth-first from the reference camera, which finds the
    shortest chain to each camera and therefore accumulates the least stereo
    error.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        camera_ids (Sequence[int]): Cameras to resolve; the first is the reference.

    Returns:
        dict: ``{camera_id: (R, T)}`` with ``p_cam = R @ p_ref + T``.

    Raises:
        FileNotFoundError: if no chain of stereo results reaches some camera.
    """
    camera_ids = list(camera_ids)
    reference = camera_ids[0]

    # Collect the available links, in both directions.
    edges = {}
    for cam_a in camera_ids:
        for cam_b in camera_ids:
            if cam_a == cam_b:
                continue
            path = cam_to_cam_path(config_path, cam_a, cam_b)
            if path is None:
                continue
            rotation, translation = load_cam_to_cam(path)
            edges.setdefault(cam_a, {})[cam_b] = (rotation, translation)
            # p_a = R^T (p_b - T)
            edges.setdefault(cam_b, {})[cam_a] = (rotation.T, -rotation.T @ translation)

    poses = {reference: (np.eye(3), np.zeros(3))}
    queue = deque([reference])
    while queue:
        current = queue.popleft()
        rotation_current, translation_current = poses[current]
        for neighbour, (rotation, translation) in edges.get(current, {}).items():
            if neighbour in poses:
                continue
            # p_neighbour = R (R_cur p_ref + T_cur) + T
            poses[neighbour] = (rotation @ rotation_current,
                                rotation @ translation_current + translation)
            queue.append(neighbour)

    missing = [camera_id for camera_id in camera_ids if camera_id not in poses]
    if missing:
        raise FileNotFoundError(
            f"No chain of cam_to_cam results reaches camera(s) {missing} from camera "
            f"{reference} under {config_path}. A stereo result is needed for every "
            f"consecutive pair, e.g. camera_0_to_camera_2.yaml.")

    return {camera_id: poses[camera_id] for camera_id in camera_ids}


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


def describe_camera_placement(config_path, camera_ids=DEFAULT_CAMERA_IDS, calib_session=None):
    """
    Report where each camera sits in the world, to check the pose convention.

    RT-COSMIK stores the camera's pose *in the world frame*, so a pose file's
    translation is the camera's own position in the room. Printing it is the
    quickest way to catch the one mistake that produces no error message:
    storing the inverse transform, which reconstructs the subject in the wrong
    place. Positions near the origin, or at an implausible height, mean the
    stored pose is inverted.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        camera_ids (Sequence[int]): Cameras to report on.
        calib_session (str | None): Calibration session, see :func:`cam_to_world_path`.

    Returns:
        dict: ``{camera_id: position}`` in metres, for cameras that have a world
        pose. Cameras without one are omitted.
    """
    placements = {}
    for camera_id in camera_ids:
        pose_file = cam_to_world_path(config_path, camera_id, calib_session)
        if pose_file is None:
            continue
        _, translation = load_cam_to_world(pose_file)
        placements[camera_id] = np.asarray(translation, dtype=float).reshape(3)
    return placements


def load_camera_parameters(config_path, camera_ids=DEFAULT_CAMERA_IDS, calib_session=None,
                           extrinsics_source="auto"):
    """
    Load intrinsics and extrinsics for a set of cameras.

    Triangulation and fusion only need the cameras' poses *relative to each
    other*; the world frame enters separately, via
    :func:`load_world_transformation`. Those relative poses can come from either
    of the two ways a rig gets calibrated:

    ``cam_to_world``
        A world pose per camera, as a motion-capture dataset produces by fitting
        each camera to shared markers. Each camera is fitted independently, so
        error does not accumulate. Preferred when available.
    ``cam_to_cam``
        Stereo results per *pair*, as a checkerboard calibration produces. These
        are chained from the reference camera. This is the usual online case,
        where a checkerboard gives intrinsics and pairwise poses and a single
        aruco marker fixes the reference camera in the world.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        camera_ids (Sequence[int]): Cameras to load, in order. The first one is
            the reference frame.
        calib_session (str | None): Calibration session to use when a camera has
            several, see :func:`cam_to_world_path`.
        extrinsics_source (str): ``"auto"`` uses cam_to_world when every camera
            has one and falls back to chaining cam_to_cam otherwise;
            ``"cam_to_world"`` or ``"cam_to_cam"`` force one source.

    Returns:
        tuple: ``(mtxs, dists, projections, rotations, translations)``, each a
        list ordered like ``camera_ids``. ``projections[i]`` is the 3x4 matrix
        ``[R | T]`` *without* the intrinsics: reconstruction undistorts to
        normalized coordinates, so K must not be baked in.
    """
    camera_ids = list(camera_ids)

    if not camera_ids:
        raise ValueError(f"No cameras requested or found under {config_path}")

    if extrinsics_source not in ("auto", "cam_to_world", "cam_to_cam"):
        raise ValueError(
            f"extrinsics_source must be 'auto', 'cam_to_world' or 'cam_to_cam', "
            f"got {extrinsics_source!r}")

    mtxs = []
    dists = []
    for camera_id in camera_ids:
        intrinsics_file = intrinsics_path(config_path, camera_id)
        if not os.path.isfile(intrinsics_file):
            raise FileNotFoundError(f"Missing intrinsics for camera {camera_id}: {intrinsics_file}")
        K, D = load_cam_params(intrinsics_file)
        mtxs.append(np.asarray(K, dtype=float))
        dists.append(np.asarray(D, dtype=float))

    world_files = {camera_id: cam_to_world_path(config_path, camera_id, calib_session)
                   for camera_id in camera_ids}
    have_all_world = all(path is not None for path in world_files.values())

    use_world = have_all_world if extrinsics_source == "auto" else \
        extrinsics_source == "cam_to_world"

    if use_world:
        if not have_all_world:
            missing = [camera_id for camera_id, path in world_files.items() if path is None]
            raise FileNotFoundError(
                f"No cam-to-world pose for camera(s) {missing} under {config_path}")
        world_poses = []
        for camera_id in camera_ids:
            R_world, T_world = load_cam_to_world(world_files[camera_id])
            world_poses.append((orthonormalize_rotation(R_world), T_world))

        # Reference camera: p_world = R_ref @ p_ref + T_ref
        R_ref, T_ref = world_poses[0]
        relative = {}
        for camera_id, (R_cam, T_cam) in zip(camera_ids, world_poses):
            # p_cam = R_cam^T (p_world - T_cam), and p_world = R_ref p_ref + T_ref,
            # so p_cam = (R_cam^T R_ref) p_ref + R_cam^T (T_ref - T_cam).
            relative[camera_id] = (R_cam.T @ R_ref, R_cam.T @ (T_ref - T_cam))
        LOGGER.info(
            "[CAL] extrinsics from cam_to_world (%s), each camera fitted "
            "independently", "forced" if extrinsics_source == "cam_to_world"
            else "auto: every camera has one")
    else:
        LOGGER.info(
            "[CAL] extrinsics by chaining cam_to_cam from camera %d (%s); error "
            "accumulates along the chain", camera_ids[0],
            "forced" if extrinsics_source == "cam_to_cam" else
            "auto: not every camera has a cam_to_world pose")
        relative = chain_relative_poses(config_path, camera_ids)

    rotations = []
    translations = []
    projections = []
    for camera_id in camera_ids:
        rotation, translation = relative[camera_id]
        translation = np.asarray(translation, dtype=float).reshape(3, 1)
        rotations.append(rotation)
        translations.append(translation)
        projections.append(np.concatenate([rotation, translation], axis=-1))
        # Baselines are the cheapest sanity check on a calibration: if one does
        # not look like the room, nothing downstream will be right.
        if camera_id != camera_ids[0]:
            LOGGER.info("[CAL] camera %d is %.3f m from the reference camera %d",
                        camera_id, float(np.linalg.norm(translation)), camera_ids[0])

    return mtxs, dists, projections, rotations, translations


def load_world_transformation(config_path, ref_camera=DEFAULT_CAMERA_IDS[0], calib_session=None,
                              required=False):
    """
    Load the transform from the reference camera frame to the world frame.

    Only the *reference* camera needs one: every other camera's pose is relative
    to it, so a single anchor places the whole rig. That anchor can be a
    Procrustes fit to motion-capture markers, or a single aruco marker viewed by
    the reference camera -- whichever wrote the file, the convention is the same,
    ``p_world = R @ p_cam + T``.

    With no anchor at all the reference camera's own frame is used as the world
    frame. Joint angles stay valid, since they only depend on relative geometry,
    but anything expressed in room coordinates -- the free-flyer translation,
    comparisons against motion capture -- is then in camera coordinates.

    Args:
        config_path (str): Calibration root in the COMFI layout.
        ref_camera (int): Camera whose frame reconstruction outputs points in,
            i.e. the first entry of the ``camera_ids`` passed to
            :func:`load_camera_parameters`.
        calib_session (str | None): Calibration session to use when the camera
            has several, see :func:`cam_to_world_path`.
        required (bool): Raise instead of falling back to identity when the
            reference camera has no world pose.

    Returns:
        tuple: ``(world_R_cam, world_T_cam)`` with shapes (3, 3) and (3,), such
        that ``p_world = world_R_cam @ p_cam + world_T_cam``.
    """
    pose_file = cam_to_world_path(config_path, ref_camera, calib_session)
    if pose_file is None:
        if required:
            raise FileNotFoundError(
                f"No cam-to-world pose for reference camera {ref_camera} under {config_path}")
        LOGGER.warning(
            "No world pose for reference camera %d under %s; using its own frame as the "
            "world frame. Joint angles are unaffected, but positions are in camera "
            "coordinates rather than room coordinates.", ref_camera, config_path)
        return np.eye(3), np.zeros(3)
    world_R_cam, world_T_cam = load_cam_to_world(pose_file)
    return orthonormalize_rotation(world_R_cam), world_T_cam.reshape((3,))
