"""The old COSMIK front end: mmpose 2D keypoints -> DLT -> OpenCap LSTM markers.

This reproduces the architecture RT-COSMIK replaced, so the paper can compare
the two on equal terms. Everything downstream of this module -- filtering, model
calibration, IK, evaluation -- is the *current* codebase, unmodified. Only the
marker source differs, which is the whole point: with ``marker_set = "parity"``
the two arms run the same 35 markers through the same model and the same solver,
so a difference in the result is a difference in pose estimation and nothing
else.

Three things make the comparison fair rather than merely similar:

*Causality.* The LSTM sees a 30-frame window of past frames only, exactly as the
old real-time pipeline used it. Running it over a whole trial at once, the usual
offline OpenCap usage, would let the baseline see the future while NLF works
frame by frame -- a handicap in the wrong direction.

*Weighting.* mmpose emits a confidence per keypoint per view. Feeding it as
``1/score`` into the weighted DLT gives the baseline the same
uncertainty-weighted multi-view fusion NLF gets, rather than a plain DLT that
would understate it.

*Marker naming.* 29 of the LSTM's 43 markers are the same anatomical landmarks
NLF emits, so they are renamed rather than re-modelled. The 14 with no NLF
counterpart -- thigh and shank tracking clusters, and the two hip joint centres
-- are dropped, which is what the old pipeline did with them too.
"""

import logging
from collections import deque
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: Halpe26, in the column order COMFI's mmpose export uses. Verified against the
#: score files' own header, and against the indices hard-coded in the augmenter.
HALPE26 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "midHip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]

#: The augmenter's output order: the lower-body model's 35 markers followed by
#: the upper-body model's 8. Taken from the response_markers lists recorded in
#: marker_augmenter.py. Note the left foot runs toe/calc/5meta where the right
#: runs toe/5meta/calc -- the asymmetry is in the trained model, not a typo.
LSTM_OUTPUT_ORDER = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
    "r_knee_study", "r_mknee_study", "r_ankle_study", "r_mankle_study",
    "r_toe_study", "r_5meta_study", "r_calc_study",
    "L_knee_study", "L_mknee_study", "L_ankle_study", "L_mankle_study",
    "L_toe_study", "L_calc_study", "L_5meta_study",
    "r_shoulder_study", "L_shoulder_study", "C7_study",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study",
    "r_lelbow_study", "r_melbow_study", "r_lwrist_study", "r_mwrist_study",
    "L_lelbow_study", "L_melbow_study", "L_lwrist_study", "L_mwrist_study",
]

#: The 29 LSTM markers that are the same landmark as an NLF marker. The 14
#: omitted ones (thigh1-3, sh1-3 both sides, and the hip joint centres) have no
#: NLF counterpart and no frame in the human model.
LSTM_TO_NLF = {
    "r.ASIS_study": "RASI",   "L.ASIS_study": "LASI",
    "r.PSIS_study": "RPSI",   "L.PSIS_study": "LPSI",
    "C7_study": "C7",
    "r_shoulder_study": "RSHO", "L_shoulder_study": "LSHO",
    "r_lelbow_study": "RELB", "r_melbow_study": "RMELB",
    "L_lelbow_study": "LELB", "L_melbow_study": "LMELB",
    "r_lwrist_study": "RWRI", "r_mwrist_study": "RMWRI",
    "L_lwrist_study": "LWRI", "L_mwrist_study": "LMWRI",
    "r_knee_study": "RKNE",   "r_mknee_study": "RMKNE",
    "L_knee_study": "LKNE",   "L_mknee_study": "LMKNE",
    "r_ankle_study": "RANK",  "r_mankle_study": "RMANK",
    "L_ankle_study": "LANK",  "L_mankle_study": "LMANK",
    "r_toe_study": "RTOE",    "r_5meta_study": "R5MHD", "r_calc_study": "RHEE",
    "L_toe_study": "LTOE",    "L_5meta_study": "L5MHD", "L_calc_study": "LHEE",
}

#: Head markers the LSTM does not produce. The old pipeline took them straight
#: from the raw keypoints, and so do we -- without them the head segment cannot
#: be built at all and the cervical DoF join the locked set.
FACE_FROM_KEYPOINTS = {
    "Nose": "Nose", "Head": "Head",
    "REar": "REar", "LEar": "LEar", "REye": "REye", "LEye": "LEye",
}

WINDOW = 30          # frames of past context the LSTM is given, as in the old code
MIN_SCORE = 1e-3     # floor, so a zero-confidence keypoint gets weight ~0 not inf


def trial_stem(mmpose_dir):
    """The filename stem COMFI's mmpose export used for this trial.

    It is not the task directory's name -- Squatting holds squat_camera_*.csv,
    Hammering holds hitting_camera_*.csv, Picking holds crouch_object_camera_*.csv
    -- and the mapping is neither derivable nor stable enough to hard-code for 23
    tasks across 18 participants. Read it off the directory instead.
    """
    mmpose_dir = Path(mmpose_dir)
    stems = {p.name.rsplit("_camera_", 1)[0]
             for p in mmpose_dir.glob("*_camera_*.csv")}
    if not stems:
        raise FileNotFoundError(f"no *_camera_*.csv under {mmpose_dir}")
    if len(stems) > 1:
        raise ValueError(
            f"{mmpose_dir} holds several trials {sorted(stems)}; "
            f"cannot tell which one is wanted")
    return stems.pop()


#: Participants whose *mmpose 2D export* is not labelled the way the rig is.
#: COMFI's 3361 has its two stereo pairs the other way round in that export: the
#: keypoints in camera_0.csv were detected in the camera calibrated as camera_4,
#: and so on. Detected without any ground truth, by reprojection error -- 68 px as
#: labelled against 7 px swapped, where every other participant is 8 to 10 px as
#: labelled. Left uncorrected it put the whole body about a metre from where
#: mocap says it was, at 54 to 61 deg across all six tasks, while still
#: reconstructing a correctly sized person -- which is why it read as a pose
#: failure rather than a labelling one.
#:
#: THE VIDEOS ARE NOT AFFECTED. This is a defect in the 2D export alone: NLF
#: reads the videos and is correct with the labelling as it stands (12.23 deg,
#: 56 mm on 3361/Lifting), and applying this correction to it instead breaks it
#: (14.78 deg, 595 mm). So only the mmpose arm may use this.
CAMERA_ID_OVERRIDES = {
    "3361": {0: 4, 2: 6, 4: 0, 6: 2},
}


def calibration_cameras(participant, cameras):
    """Calibration ids to pair with each requested camera's *mmpose* data.

    For the mmpose arm only -- see CAMERA_ID_OVERRIDES. Returns ``cameras``
    unchanged for every participant but the mislabelled ones. Works on subsets
    too, so a two-camera run picks up the same correction.
    """
    mapping = CAMERA_ID_OVERRIDES.get(str(participant))
    if not mapping:
        return list(cameras)
    return [mapping.get(c, c) for c in cameras]


def load_trial(mmpose_dir, task, cameras):
    """Read one trial's per-camera 2D keypoints and confidences.

    The keypoint files carry no header: column 0 is the frame's mean score and
    the remaining 52 are x,y per Halpe26 joint. The score files do have a header,
    and their frame column is 1-based.

    Returns:
        (keypoints, scores): ``(F, C, 26, 2)`` pixels and ``(F, C, 26)``
        confidences, trimmed to the shortest camera.
    """
    mmpose_dir = Path(mmpose_dir)
    stem = trial_stem(mmpose_dir)
    kpts, scores = [], []
    for cam in cameras:
        kp_path = mmpose_dir / f"{stem}_camera_{cam}.csv"
        sc_path = mmpose_dir / f"{stem}_scores_{cam}.csv"
        if not kp_path.exists():
            raise FileNotFoundError(f"no mmpose keypoints for camera {cam}: {kp_path}")
        if not sc_path.exists():
            raise FileNotFoundError(f"no mmpose scores for camera {cam}: {sc_path}")
        raw = np.loadtxt(kp_path, delimiter=",", ndmin=2)
        if raw.shape[1] != 1 + 2 * len(HALPE26):
            raise ValueError(
                f"{kp_path} has {raw.shape[1]} columns, expected "
                f"{1 + 2*len(HALPE26)} (mean score + 26 xy pairs)")
        kpts.append(raw[:, 1:].reshape(-1, len(HALPE26), 2))

        sc = np.genfromtxt(sc_path, delimiter=",", names=True)
        cols = [f"{n}_score" for n in HALPE26]
        missing = [c for c in cols if c not in sc.dtype.names]
        if missing:
            raise ValueError(f"{sc_path} is missing score columns: {missing}")
        scores.append(np.stack([sc[c] for c in cols], axis=1))

    frames = min(min(k.shape[0] for k in kpts), min(s.shape[0] for s in scores))
    keypoints = np.stack([k[:frames] for k in kpts], axis=1)
    confidences = np.stack([s[:frames] for s in scores], axis=1)
    return keypoints, confidences


class MmposeMarkerSource:
    """Turn one trial's mmpose output into the marker dicts the solver wants.

    Iterating yields ``(frame_index, mks_dict)`` with the 35 parity markers in
    world coordinates: 29 from the LSTM augmenter, 6 taken from the triangulated
    face keypoints.

    Args:
        keypoints: ``(F, C, 26, 2)`` pixel keypoints.
        confidences: ``(F, C, 26)`` mmpose confidences.
        mtxs, dists, projections: camera parameters, as
            ``load_camera_parameters`` returns them.
        world_R, world_T: reference-camera-to-world transform.
        models: the loaded LSTM sessions, from ``augmenter.loadModel``.
        augmenter_dir: where those models live; the augmenter re-reads its
            normalisation statistics from there.
        height, mass: the subject's, from the dataset metadata. The augmenter
            was trained with both as explicit features.
        iir: optional filter applied to the triangulated keypoints before the
            LSTM sees them, matching how the baseline was built and validated.
        buffer_len: how many frames the filter is run over, ``settings.N``, so
            the smoothing procedure matches the NLF arm's.
    """

    def __init__(self, keypoints, confidences, mtxs, dists, projections,
                 world_R, world_T, models, augmenter_dir, height, mass,
                 iir=None, buffer_len=10, logger=None, depth_aware=False):
        self.keypoints = keypoints
        self.confidences = confidences
        self.mtxs = mtxs
        self.dists = dists
        self.projections = projections
        self.world_R = np.asarray(world_R)
        self.world_T = np.asarray(world_T)
        self.models = models
        self.augmenter_dir = str(augmenter_dir)
        self.height = float(height)
        self.mass = float(mass)
        self.iir = iir
        self.buffer_len = buffer_len
        self.logger = logger or LOGGER
        self.depth_aware = depth_aware
        self._centres = None
        self._previous = None   # last frame's points, reference frame

    def __len__(self):
        return self.keypoints.shape[0]

    def triangulate(self, frame):
        """Weighted DLT for one frame, in world coordinates.

        The confidence enters as ``sigma = 1/score`` so the DLT's ``1/sigma**p``
        weighting becomes ``score**p`` -- the same shape of multi-view weighting
        NLF's per-joint uncertainty gives, from the quantity mmpose actually
        provides.
        """
        from rtcosmik.triangulation.triangulation import (
            camera_centres, distance_scaled_uncertainties, triangulate_points)

        views = [self.keypoints[frame, c] for c in range(self.keypoints.shape[1])]
        sigma = 1.0 / np.clip(self.confidences[frame], MIN_SCORE, None)

        if self.depth_aware and self._previous is not None:
            # Range predicts a view's reliability independently of confidence.
            # The distances come from the previous frame's solve, so this stays
            # causal -- at 40 Hz the body has moved millimetres, far less than
            # the metres that separate the cameras.
            if self._centres is None:
                self._centres = camera_centres(self.projections)
            sigma = distance_scaled_uncertainties(sigma, self._previous,
                                                  self._centres)

        p3d = triangulate_points(views, self.mtxs, self.dists, self.projections,
                                 uncertainties=sigma)
        self._previous = p3d
        return p3d @ self.world_R.T + self.world_T

    def __iter__(self):
        from rtcosmik.augmenter.marker_augmenter import augmentTRC

        window = deque(maxlen=WINDOW)
        smooth = deque(maxlen=self.buffer_len)
        n_kp = len(HALPE26)

        for frame in range(len(self)):
            p3d = self.triangulate(frame)

            # Mirror the NLF arm exactly: hold a buffer of buffer_len frames,
            # filter the buffer, keep its last sample. The first frame seeds the
            # buffer so filtering can start immediately rather than after a
            # silent warm-up that would shift the two arms out of step.
            if not smooth:
                for _ in range(self.buffer_len):
                    smooth.append(p3d)
            else:
                smooth.append(p3d)
            if self.iir is not None:
                block = np.asarray(smooth).reshape(self.buffer_len, 3 * n_kp)
                p3d = self.iir.filter(block).reshape(self.buffer_len, n_kp, 3)[-1]

            if not window:
                for _ in range(WINDOW):
                    window.append(p3d)
            else:
                window.append(p3d)

            augmented = augmentTRC(
                np.asarray(window), subject_mass=self.mass,
                subject_height=self.height, models=self.models,
                augmenterDir=self.augmenter_dir, augmenter_model="v0.3")
            augmented = np.asarray(augmented).reshape(len(LSTM_OUTPUT_ORDER), 3)

            mks = {LSTM_TO_NLF[name]: augmented[i]
                   for i, name in enumerate(LSTM_OUTPUT_ORDER)
                   if name in LSTM_TO_NLF}
            for nlf_name, kp_name in FACE_FROM_KEYPOINTS.items():
                mks[nlf_name] = p3d[HALPE26.index(kp_name)]
            yield frame, mks


def build_source(dataset, participant, task, cameras, settings, logger=None,
                 depth_aware=False):
    """Assemble the marker source for one trial, with its subject metadata.

    Returns ``(source, meta)``. Split out from the driver so a sweep can reuse
    one loaded set of LSTM sessions across hundreds of trials instead of paying
    the onnxruntime startup for each.
    """
    import yaml
    from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                           load_world_transformation)
    from rtcosmik.filtering.iir import IIR

    root = Path(dataset)
    meta = yaml.safe_load(
        (root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    calib = calibration_cameras(participant, cameras)
    mtxs, dists, projections, _, _ = load_camera_parameters(cam_dir, calib)
    world_R, world_T = load_world_transformation(cam_dir, calib[0])

    # The keypoint files are read by their own ids; only the calibration moves.
    keypoints, confidences = load_trial(
        root / "mmpose" / "output" / participant / task, task, cameras)

    iir = IIR(num_channel=3 * len(HALPE26), sampling_frequency=settings.fs)
    iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                   filter_type=settings.filter_type)

    source = MmposeMarkerSource(
        keypoints, confidences, mtxs, dists, projections, world_R, world_T,
        load_models(), augmenter_dir(), meta["height"], meta["weight"],
        iir=iir, buffer_len=settings.N, logger=logger, depth_aware=depth_aware)
    return source, meta


def augmenter_dir():
    """Where the LSTM sessions and their normalisation statistics live."""
    return Path(__file__).resolve().parents[1] / "augmenter" / "augmentation_model"


_MODELS = None


def load_models():
    """The LSTM sessions, loaded once per process."""
    global _MODELS
    if _MODELS is None:
        from rtcosmik.augmenter.marker_augmenter import loadModel
        _MODELS = loadModel(augmenterDir=str(augmenter_dir()),
                            augmenterModelName="LSTM", augmenter_model="v0.3")
    return _MODELS

def apply_marker_set(settings, name):
    """Switch a loaded settings namespace to another marker set.

    ``Settings._apply_marker_set`` cannot be used here: the config loader turns
    the dataclass into a SimpleNamespace and drops its methods, so a study that
    needs a different marker set than settings.py declares has to re-derive it.
    The pristine lists and the dropped-marker constants come from settings.py,
    which stays the single source of truth for what each set contains.
    """
    settings.marker_names = list(settings.full_marker_names)
    settings.keys_to_track_list = list(settings.full_keys_to_track)
    settings.nlf_indices = list(settings.full_nlf_indices)
    settings.marker_set = name
    if name == "nlf":
        settings.locked_joints = []
        return settings
    if name not in ("parity", "mocap"):
        raise ValueError(f"unknown marker set {name!r}")
    dropped = set(settings.PARITY_DROPPED_MARKERS)
    if name == "mocap":
        dropped |= set(settings.MOCAP_DROPPED_MARKERS)
    keep = [i for i, m in enumerate(settings.marker_names) if m not in dropped]
    settings.nlf_indices = [settings.nlf_indices[i] for i in keep]
    settings.marker_names = [settings.marker_names[i] for i in keep]
    settings.keys_to_track_list = [k for k in settings.keys_to_track_list
                                   if k not in dropped]
    settings.locked_joints = list(settings.PARITY_LOCKED_JOINTS)
    return settings
