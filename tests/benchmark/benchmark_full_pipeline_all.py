import os
import sys
import time
import math
import csv
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import pinocchio as pin
import yaml

# Adjust Python path to find src.rtcosmik.* (adapt the relative path if needed)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")),
)

from settings import Settings
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from rtcosmik.pose_estimator.pose_estimator import BatchPoseTrackerEstimator
from rtcosmik.filtering.iir import IIR
from rtcosmik.ik.ik import RT_IK, RT_SWIKA
from rtcosmik.human_model.urdf_model import Robot, scale_human_model, mks_registration

# Path to rt-cosmik repo root (adapt if different on your machine)
rt_cosmik_path = "/root/workspace/ros_ws/src/rt-cosmik/"


# ---------------------------------------------------------------------------
# Simple running stats helper
# ---------------------------------------------------------------------------

class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, x: float):
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def extend(self, other: "RunningStats"):
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq

    @property
    def mean(self):
        return self.sum / self.n if self.n > 0 else float("nan")

    @property
    def std(self):
        if self.n <= 1:
            return float("nan")
        m = self.mean
        var = self.sumsq / self.n - m * m
        return math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# COMFI camera parameter loaders
# ---------------------------------------------------------------------------

def _load_intrinsics_yaml(path):
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
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def _load_extrinsics_R_T_yaml(path):
    """
    Load an extrinsic transform (R, T) from either of the two formats you have:

    1) OpenCV-style calibration file (e.g. camera_0_to_camera_2.yaml):
         %YAML:1.0
         ---
         K1: !!opencv-matrix
         ...
         R:  !!opencv-matrix
         T:  !!opencv-matrix

    2) COMFI-style world extrinsics (e.g. camera_0_extrinsics.yaml):
         camera_extrinsics:
           frame_from: camera_0
           frame_to: world
           rotation_matrix: [[...], [...], [...]]
           translation_vector: [tx, ty, tz]
           ...

    Returns:
        R (np.ndarray) shape (3, 3)
        T (np.ndarray) shape (3,)
    """

    # Peek at first line to detect OpenCV vs plain YAML
    with open(path, "r") as f:
        first_line = f.readline()

    # ------------------------------------------------------------------
    # Case 1: OpenCV FileStorage (camera_0_to_camera_2.yaml)
    # ------------------------------------------------------------------
    if first_line.lstrip().startswith("%YAML"):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if fs is None or not fs.isOpened():
            raise ValueError(f"Failed to open OpenCV FileStorage for {path}")

        try:
            R_node = fs.getNode("R")
            T_node = fs.getNode("T")
            if R_node.empty() or T_node.empty():
                raise ValueError(f"Could not find R/T nodes in {path}")

            R = R_node.mat()
            T = T_node.mat()
        finally:
            fs.release()

        R = np.array(R, dtype=float)
        T = np.array(T, dtype=float).reshape(3,)
        if R.shape != (3, 3):
            raise ValueError(f"Unexpected R shape {R.shape} in {path}")
        return R, T

    # ------------------------------------------------------------------
    # Case 2: COMFI-style YAML (camera_0_extrinsics.yaml)
    # ------------------------------------------------------------------
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # sometimes everything is nested under 'camera_extrinsics'
    if "camera_extrinsics" in data:
        extr = data["camera_extrinsics"]
    else:
        extr = data

    if "rotation_matrix" not in extr or "translation_vector" not in extr:
        raise ValueError(
            f"Cannot find rotation_matrix / translation_vector in {path}"
        )

    R = np.array(extr["rotation_matrix"], dtype=float).reshape(3, 3)
    T = np.array(extr["translation_vector"], dtype=float).reshape(3,)

    return R, T


def load_comfi_world_transformation(cam_params_participant_dir):
    """
    COMFI layout:
      COMFI/cam_params/<pid>/extrinsics/cam_to_world/camera_0/camera_0_extrinsics.yaml

    We assume this file encodes CAM wrt WORLD (R_cam_world, T_cam_world),
    and we invert to get WORLD wrt CAM.
    """
    extr_path = os.path.join(
        cam_params_participant_dir,
        "extrinsics",
        "cam_to_world",
        "camera_0",
        "camera_0_extrinsics.yaml",
    )
    cam_R_world, cam_T_world = _load_extrinsics_R_T_yaml(extr_path)

    world_R_cam = cam_R_world.T
    world_T_cam = -world_R_cam @ cam_T_world
    return world_R_cam, world_T_cam


def load_comfi_camera_parameters(cam_params_participant_dir):
    """
    COMFI layout:

      intrinsics:
        COMFI/cam_params/<pid>/intrinsics/camera_0_intrinsics.yaml
        COMFI/cam_params/<pid>/intrinsics/camera_2_intrinsics.yaml

      relative extrinsics:
        COMFI/cam_params/<pid>/extrinsics/cam_to_cam/camera_0_to_camera_2.yaml

    Reference frame = camera_0.
    """
    intr_dir = os.path.join(cam_params_participant_dir, "intrinsics")
    K0, D0 = _load_intrinsics_yaml(os.path.join(intr_dir, "camera_0_intrinsics.yaml"))
    K2, D2 = _load_intrinsics_yaml(os.path.join(intr_dir, "camera_2_intrinsics.yaml"))

    rel_extr_path = os.path.join(
        cam_params_participant_dir,
        "extrinsics",
        "cam_to_cam",
        "camera_0_to_camera_2.yaml",
    )
    R_0_to_2, T_0_to_2 = _load_extrinsics_R_T_yaml(rel_extr_path)

    R0 = np.eye(3)
    T0 = np.zeros(3, dtype=float)
    R2 = R_0_to_2
    T2 = T_0_to_2

    mtxs = [K0, K2]
    dists = [D0, D2]
    rotations = [R0, R2]
    translations = [T0, T2]

    projections = []
    for K, R, T in zip(mtxs, rotations, translations):
        RT = np.hstack([R, T.reshape(3, 1)])
        P = K @ RT
        projections.append(P)

    return mtxs, dists, projections, rotations, translations


# ---------------------------------------------------------------------------
# Full pipeline benchmarker (2 cameras: 0 and 2)
# ---------------------------------------------------------------------------

class FullPipelineBenchmarker:
    """
    Sequential version of your PipelineProcess, but operating on COMFI videos
    instead of shared-memory camera buffers, and instrumented with timers.
    """

    def __init__(self, settings: Settings, comfi_root: str, ik_type: str, camera_ids=(0, 2)):
        self.settings = settings
        self.comfi_root = comfi_root
        self.videos_root = os.path.join(comfi_root, "videos")
        self.cam_params_root = os.path.join(comfi_root, "cam_params")
        self.camera_ids = camera_ids

        self.fs = settings.fs
        self.subject_mass = settings.human_mass
        self.subject_height = settings.human_height
        self.keypoints_names = settings.keypoints_names
        self.marker_names = settings.marker_names
        self.dt = settings.dt
        self.order = settings.order
        self.cutoff_freq = settings.cutoff_freq
        self.filter_type = settings.filter_type
        self.keys_to_track_list = settings.keys_to_track_list

        self.omega = {}
        for key in self.keys_to_track_list:
            self.omega[key] = 1

        self.ik_type = ik_type  # explicit IK type ('sbs' or 'mhe')
        self.ik_code = settings.ik_code
        self.cost_weights = settings.cost_weights
        self.N = settings.N

        self.buffer_max_len = 30
        self.num_cameras = len(camera_ids)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate HPE tracker (batched over cameras) and warm it up
        self.tracker = BatchPoseTrackerEstimator(
            self.num_cameras,
            settings.det_model_path,
            settings.pose_model_path,
            device=self.device,
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        _ = self.tracker.estimate([dummy_frame for _ in range(self.num_cameras)])

        # Load augmenter model once
        self.warmed_augmenter_model = loadModel(
            augmenterDir=settings.augmenter_path,
            augmenterModelName="LSTM",
            augmenter_model="v0.3",
        )

    def _setup_for_participant(self, participant_id: str):
        """
        Load camera parameters for this participant and (re)initialize
        filtering, buffers, IK state, etc.
        """
        cam_params_participant_dir = os.path.join(self.cam_params_root, participant_id)
        if not os.path.isdir(cam_params_participant_dir):
            raise RuntimeError(
                f"Missing cam_params for participant {participant_id}: {cam_params_participant_dir}"
            )

        # Camera parameters (COMFI version of load_camera_parameters / load_world_transformation)
        (
            self.mtxs,
            self.dists,
            self.projections,
            self.rotations,
            self.translations,
        ) = load_comfi_camera_parameters(cam_params_participant_dir)

        self.world_R1_cam, self.world_T1_cam = load_comfi_world_transformation(
            cam_params_participant_dir
        )

        # IIR filter setup
        num_channel = 3 * len(self.keypoints_names)
        self.iir_filter = IIR(num_channel=num_channel, sampling_frequency=self.fs)
        self.iir_filter.add_filter(
            order=self.order, cutoff=self.cutoff_freq, filter_type=self.filter_type
        )

        # Buffers & IK state
        self.keypoints_buffer = deque(maxlen=self.buffer_max_len)
        self.first_sample = True

        self.human = None
        self.human_model = None
        self.human_data = None
        self.human_collision_model = None
        self.human_visual_model = None

        self.ik_class = None
        self.deque_lstm_dict = None
        self.x_array = None
        self.u_array = None

    def _build_video_paths(self, participant_id: str, task: str):
        base_dir = os.path.join(self.videos_root, participant_id, task)
        paths = []
        for cam in self.camera_ids:
            p = os.path.join(base_dir, f"camera_{cam}.mp4")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing video for participant {participant_id}, task {task}: {p}"
                )
            paths.append(p)
        return paths

    def benchmark_trial(self, participant_id: str, task: str):
        """
        Run the full pipeline on a given (participant, task) trial and return
        per-component RunningStats per frame (excluding the first_sample frame).
        """
        self._setup_for_participant(participant_id)
        video_paths = self._build_video_paths(participant_id, task)

        caps = [cv2.VideoCapture(p) for p in video_paths]
        for p, cap in zip(video_paths, caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {p}")

        stats = {
            "hpe": RunningStats(),
            "tri_filt": RunningStats(),
            "augment": RunningStats(),
            "ik": RunningStats(),
            "pipeline": RunningStats(),
        }

        try:
            while True:
                # Read one frame per camera
                frames = []
                for cap in caps:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        frames = []
                        break
                    frames.append(frame)

                if len(frames) != self.num_cameras:
                    break  # end of at least one video

                # Remember whether this iteration is the "first_sample" one
                was_first_sample = self.first_sample

                # Start full pipeline timing (exclude disk I/O)
                t_pipe_start = time.perf_counter()

                # ------------------------------------------------------------------
                # 1) HPE (pose tracker)
                # ------------------------------------------------------------------
                t_hpe_start = time.perf_counter()
                results = self.tracker.estimate(frames)
                t_hpe_end = time.perf_counter()
                t_hpe = t_hpe_end - t_hpe_start

                # Extract keypoints per camera
                keypoints_list = []
                for res in results:
                    keypoints, bboxes, _ = res
                    keypoints = (keypoints[..., :2]).astype(float)
                    if keypoints.size == 0 or keypoints.flatten().shape != (52,):
                        continue
                    keypoints_list.append(keypoints.reshape((26, 2)).flatten())

                if len(keypoints_list) != self.num_cameras:
                    # No valid detection on all cameras; skip this frame
                    t_pipe_end = time.perf_counter()
                    continue

                # ------------------------------------------------------------------
                # 2) Triangulation + world transform + filtering
                # ------------------------------------------------------------------
                t_tri_start = time.perf_counter()
                keypoints_in_cam = triangulate_points(
                    keypoints_list, self.mtxs, self.dists, self.projections
                )
                keypoints_in_world = np.array(
                    [self.world_R1_cam @ point + self.world_T1_cam for point in keypoints_in_cam]
                )

                if self.first_sample:
                    # Add the first frame buffer_max_len times to initialize
                    for _ in range(self.buffer_max_len):
                        self.keypoints_buffer.append(keypoints_in_world)
                else:
                    self.keypoints_buffer.append(keypoints_in_world)

                if len(self.keypoints_buffer) < self.buffer_max_len:
                    # Not enough history yet
                    t_tri_end = time.perf_counter()
                    t_pipe_end = time.perf_counter()
                    continue

                keypoints_buffer_array = np.array(self.keypoints_buffer)
                filtered_keypoints_buffer = self.iir_filter.filter(
                    np.reshape(
                        keypoints_buffer_array,
                        (self.buffer_max_len, 3 * len(self.keypoints_names)),
                    )
                )
                filtered_keypoints_buffer = np.reshape(
                    filtered_keypoints_buffer,
                    (self.buffer_max_len, len(self.keypoints_names), 3),
                )
                t_tri_end = time.perf_counter()
                t_tri = t_tri_end - t_tri_start

                # ------------------------------------------------------------------
                # 3) Augmentation (LSTM-based marker augmentation)
                # ------------------------------------------------------------------
                t_aug_start = time.perf_counter()
                augmented_markers = augmentTRC(
                    filtered_keypoints_buffer,
                    subject_mass=self.subject_mass,
                    subject_height=self.subject_height,
                    models=self.warmed_augmenter_model,
                    augmenterDir=self.settings.augmenter_path,
                    augmenter_model="v0.3",
                )
                if len(augmented_markers) % 3 != 0:
                    raise ValueError(
                        "The length of the augmented markers list must be divisible by 3."
                    )
                augmented_markers = np.array(augmented_markers).reshape(-1, 3)
                t_aug_end = time.perf_counter()
                t_aug = t_aug_end - t_aug_start

                # Build dictionaries
                kp_dict = dict(
                    zip(self.keypoints_names, filtered_keypoints_buffer[-1])
                )
                mks_dict = dict(zip(self.marker_names, augmented_markers))

                # Adds head keypoints in LSTM output for head tracking
                keys_to_add = ["Nose", "Head", "REar", "LEar", "REye", "LEye"]
                mks_dict.update({key: kp_dict[key] for key in keys_to_add})

                # ------------------------------------------------------------------
                # 4) IK (either sbs / quadprog or mhe / SWIKA)
                # ------------------------------------------------------------------
                t_ik = 0.0
                if self.first_sample:
                    # First sample: initialize human model and IK, but do not
                    # include this iteration in timing statistics.

                    # Load & scale human model from URDF (as in PipelineProcess)
                    self.human = Robot(
                        os.path.join(rt_cosmik_path, "urdf", "human.urdf"),
                        rt_cosmik_path,
                        isFext=True,
                    )
                    self.human_model = self.human.model
                    self.human_data = self.human.data
                    self.human_collision_model = self.human.collision_model
                    self.human_visual_model = self.human.visual_model

                    # Scale the model to data
                    self.human_model = scale_human_model(
                        self.human_model,
                        mks_dict,
                        with_hand=True,
                        gender="male",
                        subject_height=1.70,
                    )
                    self.human_model = mks_registration(
                        self.human_model, mks_dict, with_hand=False
                    )
                    self.human_data = pin.Data(self.human_model)

                    if self.ik_type == "sbs":
                        q0 = pin.neutral(self.human_model)
                        self.ik_class = RT_IK(
                            self.human_model,
                            mks_dict,
                            q0,
                            self.keys_to_track_list,
                            self.dt,
                            self.omega,
                        )
                        # Use CasADi-based solver for initial pose
                        q_init = self.ik_class.solve_ik_sample_casadi()
                        self.ik_class._q0 = q_init

                    elif self.ik_type == "mhe":
                        self.ik_class = RT_SWIKA(
                            self.human_model,
                            self.keys_to_track_list,
                            self.N,
                            code=self.ik_code,
                        )
                        self.x_array = np.zeros(
                            (self.human_model.nq + self.human_model.nv, self.N)
                        )
                        self.x_array[6, :] = 1.0
                        self.u_array = np.zeros((self.human_model.nv, self.N))
                        self.deque_lstm_dict = deque(maxlen=self.N)
                        for _ in range(self.N):
                            self.deque_lstm_dict.append(mks_dict)
                        array_data = np.array(
                            [
                                np.hstack(
                                    [d[marker] for marker in self.keys_to_track_list]
                                )
                                for d in self.deque_lstm_dict
                            ]
                        ).T
                        self.x_array, self.u_array = self.ik_class.solve(
                            self.x_array,
                            self.u_array,
                            array_data,
                            self.x_array[:, -1],
                            self.cost_weights,
                            self.dt,
                        )
                    else:
                        raise ValueError(
                            "Invalid ik_type; should be 'sbs' or 'mhe'."
                        )

                    self.first_sample = False

                else:
                    t_ik_start = time.perf_counter()
                    if self.ik_type == "sbs":
                        self.ik_class._dict_m = mks_dict
                        q = self.ik_class.solve_ik_sample_quadprog()
                        self.ik_class._q0 = q
                    elif self.ik_type == "mhe":
                        self.deque_lstm_dict.append(mks_dict)
                        array_data = np.array(
                            [
                                np.hstack(
                                    [d[marker] for marker in self.keys_to_track_list]
                                )
                                for d in self.deque_lstm_dict
                            ]
                        ).T
                        self.x_array, self.u_array = self.ik_class.solve(
                            self.x_array,
                            self.u_array,
                            array_data,
                            self.x_array[:, -1],
                            self.cost_weights,
                            self.dt,
                        )
                        q = pin.neutral(self.human_model)
                        q[:] = np.array(
                            self.x_array[: self.human_model.nq, -1]
                        ).flatten()
                    else:
                        raise ValueError(
                            "Invalid ik_type; should be 'sbs' or 'mhe'."
                        )
                    t_ik_end = time.perf_counter()
                    t_ik = t_ik_end - t_ik_start

                # ------------------------------------------------------------------
                # 5) Full pipeline time
                # ------------------------------------------------------------------
                t_pipe_end = time.perf_counter()
                t_pipe = t_pipe_end - t_pipe_start

                # Accumulate stats for all frames EXCEPT the first_sample frame
                if not was_first_sample:
                    stats["hpe"].add(t_hpe)
                    stats["tri_filt"].add(t_tri)
                    stats["augment"].add(t_aug)
                    stats["ik"].add(t_ik)
                    stats["pipeline"].add(t_pipe)

        finally:
            for cap in caps:
                cap.release()

        return stats


# ---------------------------------------------------------------------------
# High-level benchmark over all participants and tasks
# ---------------------------------------------------------------------------

def find_participants(videos_root):
    participants = []
    for name in os.listdir(videos_root):
        p_dir = os.path.join(videos_root, name)
        if os.path.isdir(p_dir):
            participants.append(name)
    return sorted(participants)


def benchmark_full_pipeline_all():
    settings = Settings()

    # COMFI root can be overridden via env var
    comfi_root = os.environ.get(
        "COMFI_ROOT", "/home/msabbah/Desktop/comfi-examples/COMFI"
    )
    videos_root = os.path.join(comfi_root, "videos")

    tasks_of_interest = [
        "Lifting",
        "Screwing",
        "SideOverhead",
        "RobotPolishing",
        "RobotWelding",
        "Polishing",
    ]

    metrics_names = ["hpe", "tri_filt", "augment", "ik", "pipeline"]

    participants = find_participants(videos_root)
    print(f"Found {len(participants)} participants in {videos_root}")

    # IK types to benchmark in one run
    ik_types_to_run = ["mhe", "sbs"]

    # Global stats: global_stats[ik_type][metric]
    global_stats = {
        ik_type: {m: RunningStats() for m in metrics_names}
        for ik_type in ik_types_to_run
    }
    # Per-task: per_task_stats[ik_type][task][metric]
    per_task_stats = {
        ik_type: defaultdict(lambda: {m: RunningStats() for m in metrics_names})
        for ik_type in ik_types_to_run
    }
    # Per-participant: per_participant_stats[ik_type][pid][metric]
    per_participant_stats = {
        ik_type: defaultdict(lambda: {m: RunningStats() for m in metrics_names})
        for ik_type in ik_types_to_run
    }

    # Per-trial rows for CSV: one row per (participant, task, ik_type)
    per_trial_rows = []

    for ik_type in ik_types_to_run:
        print(f"\n###### Running benchmark for IK type: {ik_type} ######")
        settings.ik_type = ik_type  # keep settings consistent with the run
        benchmarker = FullPipelineBenchmarker(
            settings, comfi_root, ik_type=ik_type, camera_ids=(0, 2)
        )

        for pid in participants:
            print(f"\n=== Participant {pid} | IK={ik_type} ===")
            for task in tasks_of_interest:
                try:
                    stats = benchmarker.benchmark_trial(pid, task)
                except FileNotFoundError as e:
                    print(f"  [SKIP] {pid} | {task}: {e}")
                    continue
                except RuntimeError as e:
                    print(f"  [ERROR] {pid} | {task}: {e}")
                    continue

                # Number of valid frames for this trial (after first_sample)
                n_frames = stats["pipeline"].n
                if n_frames == 0:
                    print(
                        f"  [WARN] {pid} | {task} | IK={ik_type}: "
                        "no valid frames after initialization."
                    )
                    continue

                print(
                    f"  -> {pid} | {task} | IK={ik_type}: frames={n_frames}, "
                    + ", ".join(
                        f"{m}_mean={stats[m].mean*1000:.3f} ms, "
                        f"{m}_std={stats[m].std*1000:.3f} ms"
                        for m in metrics_names
                    )
                )

                # Store per-trial stats for CSV
                row = {
                    "participant": pid,
                    "task": task,
                    "ik_type": ik_type,
                    "frames": n_frames,
                }
                for m in metrics_names:
                    row[f"{m}_mean_ms"] = stats[m].mean * 1000.0
                    row[f"{m}_std_ms"] = stats[m].std * 1000.0
                per_trial_rows.append(row)

                # Aggregate into higher-level stats
                for m in metrics_names:
                    global_stats[ik_type][m].extend(stats[m])
                    per_task_stats[ik_type][task][m].extend(stats[m])
                    per_participant_stats[ik_type][pid][m].extend(stats[m])

    # ----------------------------------------------------------------------
    # Per-task aggregated stats
    # ----------------------------------------------------------------------
    print("\n=== Per-task aggregated statistics (all participants) ===")
    for ik_type in ik_types_to_run:
        print(f"\n-- IK type: {ik_type} --")
        for task in tasks_of_interest:
            mstats = per_task_stats[ik_type][task]
            n_frames = mstats["pipeline"].n
            if n_frames == 0:
                continue
            print(f"Task {task}: frames={n_frames}")
            for m in metrics_names:
                print(
                    f"  {m}: mean={mstats[m].mean*1000:.3f} ms, "
                    f"std={mstats[m].std*1000:.3f} ms"
                )

    # ----------------------------------------------------------------------
    # Per-participant aggregated stats
    # ----------------------------------------------------------------------
    print("\n=== Per-participant aggregated statistics (all tasks) ===")
    for ik_type in ik_types_to_run:
        print(f"\n-- IK type: {ik_type} --")
        for pid, mstats in per_participant_stats[ik_type].items():
            n_frames = mstats["pipeline"].n
            if n_frames == 0:
                continue
            print(f"Participant {pid}: frames={n_frames}")
            for m in metrics_names:
                print(
                    f"  {m}: mean={mstats[m].mean*1000:.3f} ms, "
                    f"std={mstats[m].std*1000:.3f} ms"
                )

    # ----------------------------------------------------------------------
    # Global stats
    # ----------------------------------------------------------------------
    print("\n=== Global aggregated statistics (all participants, all tasks) ===")
    for ik_type in ik_types_to_run:
        print(f"\n-- IK type: {ik_type} --")
        n_frames_global = global_stats[ik_type]["pipeline"].n
        if n_frames_global > 0:
            print(f"Total frames={n_frames_global}")
            for m in metrics_names:
                print(
                    f"  {m}: mean={global_stats[ik_type][m].mean*1000:.3f} ms, "
                    f"std={global_stats[ik_type][m].std*1000:.3f} ms"
                )
        else:
            print("No frames processed in global statistics.")

    # ----------------------------------------------------------------------
    # Save detailed per-trial stats to CSV
    # ----------------------------------------------------------------------
    csv_path = "benchmark_full_pipeline_all_results.csv"
    fieldnames = ["participant", "task", "ik_type", "frames"] + [
        f"{m}_{stat}_ms" for m in metrics_names for stat in ("mean", "std")
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_trial_rows:
            writer.writerow(row)

    print(f"\nDetailed per-trial results written to: {csv_path}")


if __name__ == "__main__":
    benchmark_full_pipeline_all()
