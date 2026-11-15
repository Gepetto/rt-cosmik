"""OpenCap LabValidation: its trials, cameras, mocap markers and released kinematics.

Layout, time base and marker mapping are described in docs/research/opencap-evaluation.md.
"""
import copy
import pickle
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np

from rtcosmik.folds import EXTERNAL_PARTICIPANTS

SUBJECTS = EXTERNAL_PARTICIPANTS          # subject6 has no videos
STATIC = "static1"
CAMERAS = (0, 1, 2, 3, 4)                 # -70, -45, 0, +45, +70 degrees
#: Arm tag -> cameras. Two and three cameras are OpenCap's own setups.
ARMS = {"nlf_5cam": (0, 1, 2, 3, 4), "nlf_3cam": (1, 2, 3), "nlf_2cam": (1, 3)}
REFERENCE_SET = "opencap_mocap"
CAMERA_FILE = "cameraIntrinsicsExtrinsics.pickle"
MM_TO_M = 1e-3
#: Both arms run in a z-up world, as on COMFI: the model calibration converges
#: to a folded posture in the checkerboard's own frame (y down).
BOARD_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])    # board: x right, y down, z into the wall
LAB_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])      # mocap: y up
#: A standing calibration posture flexes hips and knees less than this (deg).
STANDING_FLEXION_DEG = 30.0

#: Activity -> prefix of its trial names.
ACTIVITIES = {"squat": "squats", "sit_to_stand": "STS", "drop_jump": "DJ", "walking": "walking"}

#: Our mocap marker -> OpenCap's mocap marker.
MARKERS = {
    "RASI": "r.ASIS", "LASI": "L.ASIS", "RPSI": "r.PSIS", "LPSI": "L.PSIS",
    "C7": "C7", "RSHO": "R_Shoulder", "LSHO": "L_Shoulder",
    "RELB": "R_elbow_lat", "LELB": "L_elbow_lat", "RMELB": "R_elbow_med", "LMELB": "L_elbow_med",
    "RWRI": "R_wrist_radius", "LWRI": "L_wrist_radius",
    "RMWRI": "R_wrist_ulna", "LMWRI": "L_wrist_ulna",
    "RKNE": "r_knee", "LKNE": "L_knee", "RMKNE": "r_mknee", "LMKNE": "L_mknee",
    "RANK": "r_ankle", "LANK": "L_ankle", "RMANK": "r_mankle", "LMANK": "L_mankle",
    "R5MHD": "r_5meta", "L5MHD": "L_5meta", "RTOE": "r_toe", "LTOE": "L_toe",
    "RHEE": "r_calc", "LHEE": "L_calc",
}
#: In the static trial only: they scale the model and build its segment frames.
STATIC_ONLY = ("RMKNE", "LMKNE", "RMANK", "LMANK")

#: Joint columns nobody can score here: the reference has no head markers.
UNSCORED = ("Cervical_Flexion_Extension[rad]", "Cervical_Lateral_Bending[rad]",
            "Cervical_Internal_External_Rotation[rad]")

#: OpenCap coordinate -> our joint column, per side; "{s}" is r/l, "{S}" Right/Left.
MATCHED_SIDED = {
    "hip_flexion_{s}": "{S}_Hip_Flexion_Extension",
    "hip_adduction_{s}": "{S}_Hip_Abduction_Adduction",
    "hip_rotation_{s}": "{S}_Hip_Internal_External_Rotation",
    "knee_angle_{s}": "{S}_Knee_Flexion_Extension",
    "ankle_angle_{s}": "{S}_Ankle_Plantarflexion_Dorsiflexion",
    "arm_flex_{s}": "{S}_Shoulder_Flexion_Extension",
    "arm_add_{s}": "{S}_Shoulder_Abduction_Adduction",
    "arm_rot_{s}": "{S}_Shoulder_Internal_External_Rotation",
    "elbow_flex_{s}": "{S}_Elbow_Flexion_Extension",
}
MATCHED_CENTRAL = {"lumbar_extension": "Lumbar_Flexion_Extension",
                   "lumbar_bending": "Lumbar_Lateral_Bending"}
#: The rotations OpenCap's validation averages over (pelvis, lumbar, lower limbs).
RELEASED_LOWER_BODY = ("pelvis_tilt", "pelvis_list", "pelvis_rotation",
                       "lumbar_extension", "lumbar_bending", "lumbar_rotation") + tuple(
    f"{name}_{side}" for side in "rl" for name in
    ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "subtalar_angle"))
RELEASED_UPPER_BODY = tuple(f"{name}_{side}" for side in "rl"
                            for name in ("arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup"))

Trial = namedtuple("Trial", "name session activity")


def matched_pairs():
    """``[(label, [OpenCap coordinates], [our columns])]``, both sides of a joint together."""
    pairs = [(theirs, [theirs], [ours]) for theirs, ours in MATCHED_CENTRAL.items()]
    for theirs, ours in MATCHED_SIDED.items():
        pairs.append((theirs.format(s="").rstrip("_"),
                      [theirs.format(s=s) for s in "rl"],
                      [ours.format(S=S) for S in ("Right", "Left")]))
    return pairs


def dataset_settings(settings, marker_set=None):
    """A copy of ``settings`` at the dataset's frame rate and image size."""
    from rtcosmik.marker_sets import apply_marker_set
    s = copy.deepcopy(settings)
    choice = s.opencap
    s.fs, s.dt = choice["fs"], 1 / choice["fs"]
    s.width, s.height = choice["width"], choice["height"]
    s.paper["clock_offset"] = choice["clock_offset"]
    return apply_marker_set(s, marker_set or choice["marker_set"])


def activity(trial):
    """Activity of a dynamic trial, from its name."""
    for name, prefix in ACTIVITIES.items():
        if trial.startswith(prefix):
            return name
    raise ValueError(f"{trial!r} is not a dynamic trial")


def video_path(subject_dir, session, camera, trial):
    return (Path(subject_dir) / "VideoData" / f"Session{session}" / f"Cam{camera}" / trial
            / f"{trial}_syncdWithMocap.avi")


def trc_path(subject_dir, trial):
    return Path(subject_dir) / "MarkerData" / "Mocap" / f"{trial}.trc"


def released_ik(subject_dir, trial, source="Mocap"):
    """OpenCap's joint angles: ``source`` is "Mocap" or e.g. "Video/HRNet/2-cameras"."""
    return Path(subject_dir) / "OpenSimData" / source / "IK" / f"{trial}.mot"


def session_of(subject_dir, trial):
    """The session whose five cameras recorded ``trial``, or None."""
    for session in (0, 1):
        if all(video_path(subject_dir, session, c, trial).is_file() for c in CAMERAS):
            return session
    return None


def trials(subject_dir):
    """Dynamic trials with mocap markers, OpenCap's mocap angles and five synced videos."""
    found = []
    for path in sorted((Path(subject_dir) / "MarkerData" / "Mocap").glob("*.trc")):
        session = session_of(subject_dir, path.stem)
        if path.stem == STATIC or session is None:
            continue
        if released_ik(subject_dir, path.stem).is_file():
            found.append(Trial(path.stem, session, activity(path.stem)))
    return found


class _NumpyOnly(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a camera file")
        return getattr(__import__(module, fromlist=[name]), name)


def load_camera(path):
    """One camera's intrinsics and board-to-camera pose (mm)."""
    with open(path, "rb") as handle:
        return _NumpyOnly(handle).load()


def write_calibration(subject_dir, session, out_dir, cameras=CAMERAS):
    """A session's cameras in the COMFI calibration layout; the world is the checkerboard frame, z up (m)."""
    import cv2
    import yaml
    out_dir = Path(out_dir)
    for camera in cameras:
        data = load_camera(Path(subject_dir) / "VideoData" / f"Session{session}"
                           / f"Cam{camera}" / CAMERA_FILE)
        R = np.asarray(data["rotation"], float)
        t = np.asarray(data["translation"], float).ravel() * MM_TO_M
        intrinsics = out_dir / "intrinsics" / f"camera_{camera}_intrinsics.yaml"
        intrinsics.parent.mkdir(parents=True, exist_ok=True)
        storage = cv2.FileStorage(str(intrinsics), cv2.FILE_STORAGE_WRITE)
        storage.write("K", np.asarray(data["intrinsicMat"], float))
        storage.write("D", np.asarray(data["distortion"], float).reshape(1, -1))
        storage.release()
        pose = out_dir / "extrinsics" / "cam_to_world" / f"camera_{camera}" / f"camera_{camera}_extrinsics.yaml"
        pose.parent.mkdir(parents=True, exist_ok=True)
        pose.write_text(yaml.safe_dump({"camera_extrinsics": {
            "frame_from": f"camera_{camera}", "frame_to": "checkerboard",
            "rotation_matrix": (BOARD_TO_WORLD @ R.T).tolist(),
            "translation_vector": (-BOARD_TO_WORLD @ R.T @ t).tolist()}}))
    return out_dir


def read_trc(path):
    """``(times, {marker: (F, 3)})`` in the file's units; empty cells are NaN."""
    lines = Path(path).read_text().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    width = 2 + 3 * len(names)
    rows = []
    for line in lines[5:]:
        cells = line.split("\t")
        if len(cells) < 3 or not cells[0].strip():
            continue
        cells = (cells + [""] * width)[:width]
        rows.append([float(c) if c.strip() else np.nan for c in cells])
    table = np.array(rows)
    xyz = table[:, 2:].reshape(len(table), len(names), 3)
    return table[:, 1], {name: xyz[:, i] for i, name in enumerate(names)}


def read_mot(path):
    """``(times, {coordinate: (F,)})`` of an OpenSim motion file."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "endheader") + 1
    names = lines[start].split()
    table = np.array([[float(v) for v in line.split()] for line in lines[start + 1:] if line.strip()])
    return table[:, 0], {name: table[:, i] for i, name in enumerate(names) if i}


def original_video(subject_dir, session, camera, trial):
    """The whole recording a synced clip was cut from."""
    return video_path(subject_dir, session, camera, trial).with_name(f"{trial}.avi")


def _thumbnails(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vsync", "0", "-vf",
                          "scale=90:160", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, 160, 90).astype(np.float32)


def clip_start(original, clip):
    """``(first, count)``: the clip's frames are ``first .. first + count - 1`` of the original."""
    whole, part = _thumbnails(original), _thumbnails(clip)
    difference = np.abs(whole[:len(whole) - len(part) + 1] - part[0]).mean((1, 2))
    first = int(np.argmin(difference))
    if difference[first] > 0.5 or np.abs(whole[first + len(part) - 1] - part[-1]).mean() > 0.5:
        raise RuntimeError(f"{clip} is not a run of frames of {original}")
    return first, len(part)


def read_frames(path, first, count, size_wh):
    """BGR frames ``first .. first + count - 1`` of a video; None where the video has none."""
    width, height = size_wh
    start = max(first, 0)
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vsync", "0", "-vf",
         f"select='between(n,{start},{first + count - 1})',scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    try:
        for index in range(first, first + count):
            data = b"" if index < 0 else process.stdout.read(width * height * 3)
            yield (np.frombuffer(data, np.uint8).reshape(height, width, 3)
                   if len(data) == width * height * 3 else None)
    finally:
        process.kill()
        process.wait()


def _pair_lag(a, b, max_lag):
    """Rows ``b`` is ahead of ``a`` by: ``a[r]`` and ``b[r + lag]`` show the same instant."""
    best, cost = 0, np.inf
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(-lag, 0):len(a) - max(lag, 0)]
        y = b[max(lag, 0):len(b) - max(-lag, 0)]
        ok = np.isfinite(x).all((1, 2)) & np.isfinite(y).all((1, 2))
        if ok.sum() < len(a) // 2:
            continue
        d = (x[ok] - x[ok].mean(0)) - (y[ok] - y[ok].mean(0))    # a view's depth bias is not a lag
        value = np.linalg.norm(d, axis=-1).mean()
        if value < cost:
            best, cost = lag, value
    return best


def camera_offsets(views, max_lag):
    """Whole frames each camera's rows are late by, from their world-frame landmarks (F, C, M, 3).

    Camera ``c``'s row ``r`` shows instant ``r + offsets[c]``. Every pair of
    cameras is compared, and the offsets are the least-squares fit to the pair lags.
    """
    cameras = views.shape[1]
    rows, lags = [], []
    for a in range(cameras):
        for b in range(a + 1, cameras):
            row = np.zeros(cameras)
            row[a], row[b] = 1, -1
            rows.append(row)
            lags.append(_pair_lag(views[:, a], views[:, b], max_lag))
    rows.append(np.ones(cameras))
    lags.append(0)
    offsets = np.linalg.lstsq(np.array(rows), np.array(lags, float), rcond=None)[0]
    return np.round(offsets - offsets.min()).astype(int)


def align_cameras(array, offsets):
    """Rows of ``array`` (F, C, ...) shifted per camera so that a row is one instant.

    Returns the rows every camera has, and the instant of the first of them
    counted in the rows of a camera whose offset is 0.
    """
    offsets = np.asarray(offsets)
    start = offsets.max() - offsets
    length = array.shape[0] - (offsets.max() - offsets.min())
    aligned = np.stack([array[s:s + length, c] for c, s in enumerate(start)], axis=1)
    return aligned, int(offsets.max())


def _clock_cost(rows, fs, mocap_times, mocap, offset, coverage):
    times = np.arange(len(rows)) / fs + offset
    inside = (times >= mocap_times[0]) & (times <= mocap_times[-1])
    span = min(len(rows), (mocap_times[-1] - mocap_times[0]) * fs)
    if inside.sum() < coverage * span:
        return np.inf
    source = rows[inside].reshape(-1, 3)
    target = resample(mocap_times, mocap, times[inside]).reshape(-1, 3)
    ok = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    R, t = rigid_fit(source[ok], target[ok])
    return float(np.linalg.norm(source[ok] @ R.T + t - target[ok], axis=1).mean())


def clock_offset(rows, fs, mocap_times, mocap, search, coverage=0.9, step=1 / 240):
    """``(offset, residual)``: row ``r`` of the landmarks (F, M, 3) was filmed at mocap time ``r / fs + offset``.

    The offset minimises the distance left between the landmarks and the mocap
    markers (F2, M, 3) after a rigid fit, among those that keep ``coverage`` of
    the shorter of the two inside the other.
    """
    cost = lambda offset: _clock_cost(rows, fs, mocap_times, mocap, offset, coverage)
    coarse = np.arange(search[0], search[1] + 1e-9, 1 / fs)
    best = min(coarse, key=cost)
    fine = np.arange(best - 1 / fs, best + 1 / fs + 1e-9, step)
    best = min(fine, key=cost)
    return float(best), cost(best)


def resample(times, values, at):
    """Linear interpolation of ``values`` (F, ...) at ``at``; NaN outside ``times`` or beside a gap."""
    times, values, at = np.asarray(times, float), np.asarray(values, float), np.asarray(at, float)
    right = np.clip(np.searchsorted(times, at, side="left"), 1, len(times) - 1)
    left = right - 1
    weight = (at - times[left]) / (times[right] - times[left])
    weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        out = (1 - weight) * values[left] + weight * values[right]
    out = np.where(np.isclose(weight, 1), values[right], np.where(np.isclose(weight, 0), values[left], out))
    out[(at < times[0] - 1e-9) | (at > times[-1] + 1e-9)] = np.nan
    return out


def reference_markers(path, at=None):
    """A trial's mocap markers under our names, in metres and z up, at times ``at`` (default: as sampled)."""
    times, theirs = read_trc(path)
    markers = {}
    for ours, name in MARKERS.items():
        if name in theirs:
            xyz = theirs[name] @ LAB_TO_WORLD.T * MM_TO_M
            markers[ours] = xyz if at is None else resample(times, xyz, at)
    return markers


def require_standing(joint_names, q):
    """Refuse a calibration whose posture is not a standing one: the IK settled in a wrong minimum."""
    flexion = {name: abs(np.degrees(value)) for name, value in zip(joint_names, q)
               if "Hip_Flexion" in name or "Knee_Flexion" in name}
    worst = max(flexion, key=flexion.get)
    if flexion[worst] > STANDING_FLEXION_DEG:
        raise RuntimeError(f"calibration posture has {worst} at {flexion[worst]:.0f} deg")


def window_frames(times, time_range):
    """First and last frame whose time lies in ``time_range``."""
    inside = np.flatnonzero((np.asarray(times) >= time_range[0] - 1e-6)
                            & (np.asarray(times) <= time_range[1] + 1e-6))
    return int(inside[0]), int(inside[-1])


def rigid_fit(source, target):
    """``(R, t)`` minimising ``|source @ R.T + t - target|`` over the rows finite in both."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    ok = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    source, target = source[ok], target[ok]
    a, b = source.mean(0), target.mean(0)
    U, _, Vt = np.linalg.svd((source - a).T @ (target - b))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, b - R @ a


def released_errors(video_mot, mocap_mot, time_range=None):
    """``{coordinate: {rmse, mae, frames}}`` (degrees) of one released video result against the mocap one."""
    t_video, video = read_mot(video_mot)
    t_mocap, mocap = read_mot(mocap_mot)
    key = lambda t: np.round(t * 100).astype(int)
    common, rows_video, rows_mocap = np.intersect1d(key(t_video), key(t_mocap), return_indices=True)
    if time_range is not None:
        keep = (common >= round(time_range[0] * 100)) & (common <= round(time_range[1] * 100))
        rows_video, rows_mocap = rows_video[keep], rows_mocap[keep]
    errors = {}
    for name in video:
        if name in mocap:
            d = video[name][rows_video] - mocap[name][rows_mocap]
            errors[name] = {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))),
                            "frames": int(len(d))}
    return errors


def agreement(a, b):
    """Correlation of two angle series and their RMSE once sign and mean difference are removed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    r = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) or np.inf))
    return r, float(np.sqrt(np.mean((np.sign(r or 1.0) * a - b) ** 2)))
