#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import cv2
import numpy as np
import torch
import pinocchio as pin 
from pinocchio.visualize import MeshcatVisualizer

from rtcosmik.config_loader import settings
from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.filtering.iir import IIR
from rtcosmik.human_model.model_utils import scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources
from rtcosmik.pipeline.pipeline import PipelineProcess
from rtcosmik.viewer.viewer import ViewerProcess

from multiprocessing import set_start_method
from collections import deque
import example_robot_data as robex

import logging

import ctypes
import subprocess
import json

import threading
import queue
import os

try:
    # Hard-override the process affinity mask back to all 32 cores
    os.sched_setaffinity(0, set(range(32)))
except Exception as e:
    print(f"[PRE-START WARNING] Could not force 32-core affinity: {e}", file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

LOGGER = logging.getLogger(__name__)



def _get_allowed_cpus():
    """The actual CPU IDs this process may run on, respecting cgroup/cpuset
    limits (Docker --cpus, Kubernetes resources.limits.cpu, taskset, etc).
    This is authoritative -- unlike lscpu/nproc, which can show host-wide
    topology even inside a restricted container."""
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return list(range(os.cpu_count() or 1))


def _get_physical_core_id(cpu_id):
    """Physical core ID for a logical CPU, to detect hyperthread siblings."""
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id"
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return cpu_id  # fallback: treat as its own isolated core


def _get_cpu_max_freq_khz(cpu_id):
    """Max frequency this logical CPU can reach — used to distinguish
    P-cores (high) from E-cores (lower) on hybrid Intel chips, since
    /topology/core_id alone doesn't tell you which is which."""
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq"
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0  # unknown -> treated as lowest priority


def _compute_core_pins(allowed_cpus):
    by_core = {}
    for cpu in allowed_cpus:
        core = _get_physical_core_id(cpu)
        by_core.setdefault(core, []).append(cpu)

    # Rank physical cores by max frequency, fastest first -- puts P-cores
    # ahead of E-cores on hybrid chips, so ik_worker (the heaviest, most
    # latency-sensitive single-thread workload) gets a fast core.
    def core_max_freq(cpu_list):
        return max(_get_cpu_max_freq_khz(c) for c in cpu_list)

    physical_cores = sorted(by_core.values(), key=core_max_freq, reverse=True)

    if len(physical_cores) >= 3:
        ik_cpus = physical_cores[0]                                   # fastest core, exclusive
        reader_cpus = physical_cores[-1]                               # slowest core is fine for I/O-bound reader
        gpu_cpus = [c for core in physical_cores[1:-1] for c in core]  # everything else
    elif len(physical_cores) == 2:
        ik_cpus = physical_cores[0]        # fastest core, exclusive
        reader_cpus = physical_cores[1]
        gpu_cpus = physical_cores[1]
    else:
        n = len(allowed_cpus)
        third = max(1, n // 3)
        reader_cpus = allowed_cpus[:third] or allowed_cpus[:1]
        gpu_cpus = allowed_cpus[third:2 * third] or allowed_cpus[:1]
        ik_cpus = allowed_cpus[2 * third:] or allowed_cpus[-1:]

    return {'reader': reader_cpus, 'gpu_worker': gpu_cpus, 'ik_worker': ik_cpus}


def _pin_current_thread_to_cores(core_ids):
    """Pin the calling OS thread to a specific set of CPU cores (Linux only).

    This is per-THREAD affinity, separate from OMP_NUM_THREADS/OMP_PROC_BIND env
    vars (which only govern threads spawned inside OpenMP-parallel regions).
    Used here to keep ik_worker's acados solve() off the same physical cores
    as reader/gpu_worker's CPU-side work (YOLO NMS, tensor preprocessing),
    reducing cache/scheduler contention between them.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        cpu_set_t_size = 128  # bytes, generous upper bound for CPU_SETSIZE
        mask = (ctypes.c_uint8 * cpu_set_t_size)()
        for core_id in core_ids:
            mask[core_id // 8] |= (1 << (core_id % 8))
        SYS_gettid = 186  # x86_64; use 178 on aarch64 if needed
        tid = libc.syscall(SYS_gettid)
        ret = libc.sched_setaffinity(tid, cpu_set_t_size, ctypes.byref(mask))
        if ret != 0:
            errno = ctypes.get_errno()
            LOGGER.warning(f"[WARN] sched_setaffinity failed for cores {core_ids}: errno={errno}")
        else:
            LOGGER.info(f"[INFO] Pinned thread (tid={tid}) to cores {core_ids}")
    except Exception as exc:
        LOGGER.warning(f"[WARN] Could not set CPU affinity to {core_ids}: {exc}")

# -----------------------
# Meshcat debug helpers
# -----------------------


def make_triad_geom(axis_length=0.08, linewidth=2):
    """
    RGB triad as LineSegments:
      X = red, Y = green, Z = blue
    Compatible with meshcat versions that don't have g.Axes.
    """
    if hasattr(g, "Axes"):
        return g.Axes(axis_length=axis_length)

    pts = np.array([
        [0.0, axis_length,  0.0, 0.0,       0.0, 0.0],
        [0.0, 0.0,          0.0, axis_length,0.0, 0.0],
        [0.0, 0.0,          0.0, 0.0,       0.0, axis_length],
    ], dtype=np.float32)

    cols = np.array([
        [255, 255,   0,   0,   0,   0],  # R
        [  0,   0, 255, 255,   0,   0],  # G
        [  0,   0,   0,   0, 255, 255],  # B
    ], dtype=np.uint8)

    geom = g.PointsGeometry(position=pts, color=cols)
    mat  = g.LineBasicMaterial(vertexColors=True, linewidth=linewidth)
    return g.LineSegments(geom, mat)


def make_empty_pointcloud():
    P = np.zeros((3, 0), dtype=np.float32)
    C = np.zeros((3, 0), dtype=np.uint8)

    if hasattr(g, "PointCloud"):
        return g.PointCloud(P, C)

    geom = g.PointsGeometry(position=P, color=C)
    mat = g.PointsMaterial(size=0.005, vertexColors=True)
    return g.Points(geom, mat)


def _pin_se3_to_meshcat_tf(M: pin.SE3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T

def setup_debug_visuals(
    vis,
    model: pin.Model,
    marker_names,
    triad_length=0.08,
    triad_radius=0.003,
    root="debug",
    clear_root=True,
):
    if clear_root:
        try:
            vis[root].delete()
        except Exception:
            pass

    dbg = {
        "root": root,
        "joint_entries": [],
        "marker_entries": [],
        "model_marker_path": f"{root}/model_markers",
        "missing_marker_frames": [],
    }

    triad = make_triad_geom(axis_length=triad_length, linewidth=max(1, int(triad_radius * 500)))

    for jid in range(1, model.njoints):
        jname = model.names[jid]
        path = f"{root}/joints/{jid:04d}_{jname}"
        vis[path].set_object(triad)
        dbg["joint_entries"].append((jid, path))

    for mk in marker_names:
        try:
            fid = model.getFrameId(mk)
        except Exception:
            fid = None

        if fid is None or fid < 0 or fid >= len(model.frames):
            dbg["missing_marker_frames"].append(mk)
            continue

        path = f"{root}/marker_frames/{fid:04d}_{mk}"
        vis[path].set_object(triad)
        dbg["marker_entries"].append((fid, path))

    vis[dbg["model_marker_path"]].set_object(make_empty_pointcloud())

    if dbg["missing_marker_frames"]:
        print("[DEBUG] marker frames missing in model (not registered / not added):")
        print("        ", dbg["missing_marker_frames"])

    return dbg


def update_debug_visuals(vis, model: pin.Model, data: pin.Data, q, dbg):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for jid, path in dbg.get("joint_entries", []):
        vis[path].set_transform(_pin_se3_to_meshcat_tf(data.oMi[jid]))

    marker_points = []
    for fid, path in dbg.get("marker_entries", []):
        oMf = data.oMf[fid]
        vis[path].set_transform(_pin_se3_to_meshcat_tf(oMf))
        marker_points.append(oMf.translation)

    if marker_points:
        P = np.stack(marker_points, axis=1)
        C = np.tile(np.array([[0], [255], [0]], dtype=np.uint8), (1, P.shape[1]))
        vis[dbg.get("model_marker_path", "debug/model_markers")].set_object(g.PointCloud(P, C))

# -----------------------
# Named measured markers (debug)
# -----------------------

def setup_measured_markers(vis: "meshcat.Visualizer", marker_names: Sequence[str], radius: float = 0.010, color: int = 0xff0000):
    sphere = g.Sphere(radius)
    mat = g.MeshPhongMaterial(color=color, opacity=0.9)
    for name in marker_names:
        vis[f"markers/measured/{name}"].set_object(sphere, mat)


def update_measured_markers(vis: "meshcat.Visualizer", mks_dict: dict):
    for name, p in mks_dict.items():
        try:
            T = tf.translation_matrix(np.asarray(p, dtype=float).reshape(3))
        except Exception:
            continue
        vis[f"markers/measured/{name}"].set_transform(T)

def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")
    vids = [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in [".mp4"]]
    return vids

@dataclass
class OfflineVideoSource:
    paths: List[Path]
    size_wh: Tuple[int, int]

    def __post_init__(self):
        self.caps = [cv2.VideoCapture(str(p)) for p in self.paths]
        for p, cap in zip(self.paths, self.caps):
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {p}")

    def read(self) -> Optional[List[np.ndarray]]:
        frames: List[np.ndarray] = []
        for cap in self.caps:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    return None
            W, H = self.size_wh
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        return frames

    def release(self):
        for cap in self.caps:
            cap.release()


def run_pipelined(src, est, vis, vis_markers, mtxs, dists, projections,
                   world_R1_cam, world_T1_cam, settings, total_frames, stop_event,
                   core_pins=None):
    """core_pins: optional dict like {'reader': [0], 'gpu_worker': [1,2,3],
    'ik_worker': [4,5]} to pin each worker thread to specific CPU cores.
    Pass None (default) to skip pinning entirely."""
    core_pins = core_pins or {}

    frame_q = queue.Queue(maxsize=3)
    infer_q = queue.Queue(maxsize=3)
    SENTINEL = None

    read_times, nlf_times, ik_times = [], [], []
    tri_filt_times = []           # triangulation + IIR filter time per frame
    marker_viz_times = []         # vis_markers.set_object time per frame
    solve_times = []              # ik_class.solve(...) time per frame (steady-state only)
    display_viz_times = []        # viz_human.display(q) time per frame (steady-state only)
    frame_latencies = []          # end-to-end: read-start -> ik-done, per frame
    frame_latency_idx = []        # matching frame idx for each entry in frame_latencies
    frame_start_ts = {}           # idx -> perf_counter() when reader started that frame
    frame_start_lock = threading.Lock()

    first_read_ts = [None]        # perf_counter() when the very first frame started reading
    last_done_ts = [None]         # perf_counter() when the most recent frame finished ik
    calib_done_ts = [None]        # perf_counter() when the calibration frame finished (1st completed frame)
    warm_done_ts = [None]         # perf_counter() when the first-call compile-spike frame finished (2nd completed frame)
    span_lock = threading.Lock()

    ik_class_out = [None]         # holds the constructed ik_class after calibration, for post-run diagnostics

    def reader():
        if core_pins.get('reader'):
            _pin_current_thread_to_cores(core_pins['reader'])
        idx = 0
        while not stop_event.is_set() and idx < total_frames:
            t0 = time.perf_counter()
            frames = src.read()
            if frames is None:
                break
            read_times.append((time.perf_counter() - t0) * 1000.0)
            with frame_start_lock:
                frame_start_ts[idx] = t0
            with span_lock:
                if first_read_ts[0] is None:
                    first_read_ts[0] = t0
            frame_q.put((idx, frames))
            idx += 1
        frame_q.put(SENTINEL)

    def gpu_worker():
        if core_pins.get('gpu_worker'):
            _pin_current_thread_to_cores(core_pins['gpu_worker'])
        while True:
            item = frame_q.get()
            if item is SENTINEL:
                infer_q.put(SENTINEL)
                break
            idx, frames = item
            t0 = time.perf_counter()
            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
            nlf_times.append((time.perf_counter() - t0) * 1000.0)
            infer_q.put((idx, frames, nlf_out, boxes))

    def ik_worker():
        if core_pins.get('ik_worker'):
            _pin_current_thread_to_cores(core_pins['ik_worker'])
            LOGGER.info(f"[INFO] ik_worker actual affinity after pin: {sorted(os.sched_getaffinity(0))}")
        first_sample = True
        p3d_buffer = deque(maxlen=settings.N)
        num_channel = 3 * len(settings.marker_names)
        iir_filter = IIR(num_channel=num_channel, sampling_frequency=settings.fs)
        iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                               filter_type=settings.filter_type)

        human_model = None
        human_data = None
        viz_human = None
        ik_class = None
        x_array = u_array = None
        deque_lstm_dict = None

        while True:
            item = infer_q.get()
            if item is SENTINEL:
                break
            idx, frames, nlf_out, boxes = item
            t0 = time.perf_counter()

            nlf_out_2d = nlf_out["poses2d"]
            NUM_CAMERAS = len(frames)
            if nlf_out_2d is None or len(nlf_out_2d) < NUM_CAMERAS:
                continue

            keypoints_list = [None] * NUM_CAMERAS
            valid_cam_ids = []
            for ii in range(NUM_CAMERAS):
                poses2d = nlf_out_2d[ii]
                if poses2d is None or len(poses2d) == 0 or poses2d[0] is None:
                    continue
                keypoints_list[ii] = poses2d[0].detach().float().cpu().numpy()
                valid_cam_ids.append(ii)

            if len(valid_cam_ids) < 2:
                continue

            t_tri0 = time.perf_counter()
            p3d = triangulate_points(keypoints_list=keypoints_list, mtxs=mtxs,
                                      dists=dists, projections=projections)
            p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)
            p3d_in_world = np.array([np.dot(world_R1_cam, pt) + world_T1_cam for pt in p3d_np])

            if first_sample:
                for _ in range(settings.N):
                    p3d_buffer.append(p3d_in_world)
            else:
                p3d_buffer.append(p3d_in_world)

            if len(p3d_buffer) != settings.N:
                continue

            p3d_buffer_array = np.array(p3d_buffer)
            filtered = iir_filter.filter(
                np.reshape(p3d_buffer_array, (settings.N, 3 * len(settings.marker_names)))
            )
            filtered = np.reshape(filtered, (settings.N, len(settings.marker_names), 3))
            augmented_markers = filtered[-1]
            t_tri1 = time.perf_counter()
            tri_filt_times.append((t_tri1 - t_tri0) * 1000.0)

            colors = np.zeros_like(augmented_markers.T)
            colors[0, :] = 1.0
            colors[1, :] = 0.0
            colors[2, :] = 0.0
            t_mviz0 = time.perf_counter()
            vis_markers.set_object(g.PointCloud(position=augmented_markers.T, color=colors, size=0.02))
            marker_viz_times.append((time.perf_counter() - t_mviz0) * 1000.0)

            mks_dict = dict(zip(settings.marker_names, augmented_markers))

            if first_sample:
                human = robex.human.HumanLoader(
                    height=settings.human_height,
                    weight=settings.human_weight,
                    gender=settings.human_gender
                ).robot
                human_model = human.model
                human_collision_model = human.collision_model
                human_visual_model = human.visual_model

                human_model = scale_human_model(
                    human_model, mks_dict, gender=settings.human_gender,
                    subject_height=settings.human_height
                )
                human_model = mks_registration(
                    human_model, mks_dict, gender=settings.human_gender,
                    subject_height=settings.human_height
                )

                viz_human = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
                viz_human.initViewer(vis, open=True)

                try:
                    vis["ref"].delete()
                except Exception:
                    pass
                viz_human.loadViewerModel("ref")

                viz_human.viewer["/Background"].set_property("top_color", [1, 1, 1])
                viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

                if settings.ik_type == 'sbs':
                    omega = {key: 1 for key in settings.keys_to_track_list}
                    q = pin.neutral(human_model)
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                    q = ik_class.solve_ik_sample_casadi()
                    ik_class._q0 = q
                    viz_human.display(q)

                    human_model = recalibrate_marker_frames_in_joint_space(
                        human_model, q, mks_dict, settings.marker_names
                    )
                    human_data = human_model.createData()

                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
                    LOGGER.info("[INFO] Model calibration finished, ready to process...")

                elif settings.ik_type == 'mhe':
                    x_array = np.zeros((human_model.nq + human_model.nv, settings.N))
                    x_array[6, :] = 1
                    u_array = np.zeros((human_model.nv, settings.N))
                    deque_lstm_dict = deque(maxlen=settings.N)
                    for _ in range(settings.N):
                        deque_lstm_dict.append(mks_dict)

                    omega = {key: 1 for key in settings.keys_to_track_list}
                    q = pin.neutral(human_model)
                    ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                    q = ik_class.solve_ik_sample_casadi()
                    ik_class._q0 = q
                    viz_human.display(q)

                    human_model = recalibrate_marker_frames_in_joint_space(
                        human_model, q, mks_dict, settings.marker_names
                    )
                    human_data = human_model.createData()

                    if settings.mhe_backend == 'acados':
                        ik_class = RT_SWIKA_ACADOS(
                            human_model, settings.keys_to_track_list, settings.N, settings.dt,
                            export_dir=settings.acados_export_dir,
                            acados_source_dir=settings.acados_source_dir,
                            max_iter=settings.mhe_max_iter
                        )
                    else:
                        ik_class = RT_SWIKA_FATROP(
                            human_model, settings.keys_to_track_list, settings.N, code=settings.ik_code,
                            max_iter=settings.mhe_max_iter
                        )
                    LOGGER.info("[INFO] Model calibration finished, ready to process...")
                else:
                    raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                ik_class_out[0] = ik_class
                first_sample = False

            else:
                if settings.ik_type == 'sbs':
                    t_solve0 = time.perf_counter()
                    ik_class._dict_m = mks_dict
                    q = ik_class.solve_ik_sample_quadprog()
                    ik_class._q0 = q
                    t_solve1 = time.perf_counter()
                    viz_human.display(q)
                    display_viz_times.append((time.perf_counter() - t_solve1) * 1000.0)
                    solve_times.append((t_solve1 - t_solve0) * 1000.0)
                elif settings.ik_type == 'mhe':
                    deque_lstm_dict.append(mks_dict)
                    array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list])
                                            for d in deque_lstm_dict]).T

                    t_solve0 = time.perf_counter()
                    x_array, u_array = ik_class.solve(x_array, u_array, array_data,
                                                       x_array[:, -1], settings.cost_weights, settings.dt)
                    t_solve1 = time.perf_counter()

                    q = pin.neutral(human_model)
                    q[:] = np.array(x_array[:human_model.nq, -1]).flatten()
                    viz_human.display(q)
                    display_viz_times.append((time.perf_counter() - t_solve1) * 1000.0)
                    solve_times.append((t_solve1 - t_solve0) * 1000.0)
                else:
                    raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

            ik_times.append((time.perf_counter() - t0) * 1000.0)

            with frame_start_lock:
                frame_t0 = frame_start_ts.pop(idx, None)
            done_ts = time.perf_counter()
            if frame_t0 is not None:
                frame_latencies.append((done_ts - frame_t0) * 1000.0)
                frame_latency_idx.append(idx)
            with span_lock:
                if calib_done_ts[0] is None:
                    # 1st completed frame: paid the calibration cost
                    # (human model load, meshcat init, first casadi solve).
                    calib_done_ts[0] = done_ts
                elif warm_done_ts[0] is None:
                    # 2nd completed frame: paid the first-call cost of the
                    # steady-state solve path (quadprog JIT / acados-fatrop
                    # first solve, dlopen, solver memory init, etc.).
                    warm_done_ts[0] = done_ts
                last_done_ts[0] = done_ts

    threads = [threading.Thread(target=fn, daemon=True) for fn in (reader, gpu_worker, ik_worker)]
    for t in threads: t.start()
    for t in threads: t.join()

    total_first_to_last_s = None
    if first_read_ts[0] is not None and last_done_ts[0] is not None:
        total_first_to_last_s = last_done_ts[0] - first_read_ts[0]

    steady_state_s = None
    if calib_done_ts[0] is not None and last_done_ts[0] is not None:
        steady_state_s = last_done_ts[0] - calib_done_ts[0]

    # True steady state: excludes BOTH the calibration frame AND the frame
    # that paid the first-call compile/JIT cost of the steady-state solve.
    warm_steady_state_s = None
    if warm_done_ts[0] is not None and last_done_ts[0] is not None:
        warm_steady_state_s = last_done_ts[0] - warm_done_ts[0]

    return (read_times, nlf_times, ik_times, frame_latencies, frame_latency_idx,
            total_first_to_last_s, steady_state_s, warm_steady_state_s,
            tri_filt_times, marker_viz_times, solve_times, display_viz_times,
            ik_class_out[0])


def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # PyTorch has its own CPU thread pools (intra-op / inter-op), separate from
    # OMP_NUM_THREADS/OMP_PROC_BIND env vars, which only govern OpenMP-parallel
    # regions. Left unconstrained, these can spawn multi-core thread pools during
    # YOLO NMS / tensor preprocessing that compete with ik_worker's acados solve
    # for the same physical cores. Pin both to 1 to remove that source of
    # contention; combine with core affinity pinning below for the full effect.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    W = settings.width
    H = settings.height
    mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
    world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)

    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        results_queues = create_pipeline_shared_ressources()

        camera_processes = [
            Camera(list(cameras.keys())[i], 
                camera_buffers[i], 
                camera_timestamps[i], 
                camera_locks[i], 
                frame_counters[i], 
                camera_barrier, 
                stop_event, 
                FRAME_SHAPE, 
                settings.fs, 
                settings.fourcc,)
            for i in range(NUM_CAMERAS)
        ]

        pipeline = PipelineProcess(
            settings=settings,
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            results_queues=results_queues,
            stop_event=stop_event,
            mtxs=mtxs,
            dists=dists,
            projections=projections,
            world_R1_cam=world_R1_cam,
            world_T1_cam=world_T1_cam,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
        )

        viewer= ViewerProcess(
            settings=settings,
            results_queues=results_queues,
            stop_event=stop_event,
            num_cameras=NUM_CAMERAS,
        )

        processes = camera_processes + [pipeline, viewer]

        for p in processes:
            p.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
            for process in processes:
                process.stop() if hasattr(process, 'stop') else None
                process.join(timeout=2)

    else: # offline mode

        vis = meshcat.Visualizer()
        LOGGER.info(f"[INFO] Meshcat visualizer available here: {vis.url()}")

        vis_markers = vis["markers"]

        if args.videos and len(args.videos) > 0:
            paths = [Path(v) for v in args.videos]
        else:
            paths = list_videos(Path(args.data_dir))
        if len(paths) == 0:
            raise RuntimeError(f"No videos found in {args.data_dir}")

        NUM_CAMERAS = len(paths)

        src = OfflineVideoSource(paths=paths, size_wh=(W, H))

        est = NLFEstimator(
            yolo_path=settings.yolo_path,
            nlf_path=settings.nlf_path,
            cano_path=settings.cano_path,
            image_size=(W, H),
            cam_Ks=mtxs,
            indices=settings.nlf_indices,
            conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz,
            device=settings.device,
        )

        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames',
            '-of', 'json', str(paths[0])
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        total_frames = int(data['streams'][0]['nb_frames'])
        LOGGER.info(f"[INFO] Total frames determined from ffprobe: {total_frames}")

        stop_event = threading.Event()

        # Core assignment: reserve dedicated cores for ik_worker (acados),
        # separate from reader/gpu_worker's CPU-side work (YOLO NMS, tensor
        # preprocessing). ADJUST THESE to match the actual machine -- check
        # `nproc` / `lscpu` first. This example assumes >=6 usable cores;
        # if fewer are available, shrink the sets accordingly (they must not
        # overlap for the pinning to have any effect).
        allowed_cpus = _get_allowed_cpus()
        core_pins = _compute_core_pins(allowed_cpus)
        LOGGER.info(f"[INFO] Process allowed CPUs: {allowed_cpus}")
        LOGGER.info(f"[INFO] Computed core pins: {core_pins}")

        (read_times, nlf_times, ik_times, frame_latencies, frame_latency_idx,
         total_first_to_last_s, steady_state_s, warm_steady_state_s,
         tri_filt_times, marker_viz_times, solve_times, display_viz_times,
         ik_class) = run_pipelined(
            src, est, vis, vis_markers, mtxs, dists, projections,
            world_R1_cam, world_T1_cam, settings, total_frames, stop_event,
            core_pins=core_pins
        )

        # Drop BOTH one-time-cost frames: (1) calibration, (2) first-call
        # compile/JIT spike of the steady-state solve path. Only dropping
        # one leaves the other spike sitting inside "steady state" stats.
        n_dropped = min(2, len(frame_latencies))
        dropped_idxs = frame_latency_idx[:n_dropped]
        del frame_latencies[:n_dropped]
        del frame_latency_idx[:n_dropped]
        del ik_times[:n_dropped]
        del tri_filt_times[:n_dropped]
        del marker_viz_times[:n_dropped]
        # solve_times / display_viz_times only ever contain steady-state
        # frames (calibration doesn't append to them), so only the single
        # first-call compile-spike entry needs dropping here, not two.
        if solve_times:
            solve_times.pop(0)
        if display_viz_times:
            display_viz_times.pop(0)
        if dropped_idxs:
            LOGGER.info(f"[INFO] Dropped one-time-cost frames idx={dropped_idxs} from latency/IK stats")

        n_processed = len(frame_latencies)

        print("\n--- BENCHMARK RESULTS ---")
        print(f"Total Video Frames        : {total_frames}")
        print(f"Frames fully processed    : {n_processed}")
        if read_times:
            print(f"Read:      mean {np.mean(read_times):.1f} ms | median {np.median(read_times):.1f} ms | max {np.max(read_times):.1f} ms")
        if nlf_times:
            print(f"NLF:       mean {np.mean(nlf_times):.1f} ms | median {np.median(nlf_times):.1f} ms | max {np.max(nlf_times):.1f} ms")
        if ik_times:
            print(f"IK:        mean {np.mean(ik_times):.1f} ms | median {np.median(ik_times):.1f} ms | max {np.max(ik_times):.1f} ms")

        print("\n--- IK sub-stage breakdown (steady-state frames) ---")
        if tri_filt_times:
            print(f"Triangulate+Filter: mean {np.mean(tri_filt_times):.1f} ms | median {np.median(tri_filt_times):.1f} ms | max {np.max(tri_filt_times):.1f} ms")
        if marker_viz_times:
            print(f"Marker viz (meshcat set_object): mean {np.mean(marker_viz_times):.1f} ms | median {np.median(marker_viz_times):.1f} ms | max {np.max(marker_viz_times):.1f} ms")
        if solve_times:
            print(f"Solve (ik_class.solve/quadprog): mean {np.mean(solve_times):.1f} ms | median {np.median(solve_times):.1f} ms | max {np.max(solve_times):.1f} ms")
        if display_viz_times:
            print(f"Display viz (meshcat viz_human.display): mean {np.mean(display_viz_times):.1f} ms | median {np.median(display_viz_times):.1f} ms | max {np.max(display_viz_times):.1f} ms")

        # Anchored right after calibration only (still includes the 2nd
        # frame's first-call compile spike inside the span).
        if steady_state_s is not None:
            print(f"\nPost-calibration span (includes first-call compile spike): {steady_state_s:.2f} s")

        # Anchored right after calibration AND the first-call compile spike.
        # This is the number that reflects real steady-state running speed.
        if warm_steady_state_s is not None and n_processed > 0:
            print(f"Steady-state time (init, compilation and first frame excluded): {warm_steady_state_s:.2f} s")
            print(f"Steady-state throughput: {n_processed / warm_steady_state_s:.2f} FPS")

        if settings.ik_type == 'mhe' and settings.mhe_backend == 'acados' and ik_class is not None:
            ik_class.print_core_performance_breakdown()

        src.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)