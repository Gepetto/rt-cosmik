#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir
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

from src.rtcosmik.config_loader import settings
from src.rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from src.rtcosmik.triangulation.triangulation import triangulate_points
from src.rtcosmik.filtering.iir import IIR
from src.rtcosmik.human_model.model_utils import scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space
from src.rtcosmik.ik.ik import RT_IK, RT_SWIKA
from src.rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from src.rtcosmik.camera.camera import Camera
from src.rtcosmik.utils.mp_utils import create_udp_buffer, create_camera_shared_ressources, create_pipeline_shared_ressources
from src.rtcosmik.pipeline.pipeline import PipelineProcess

from multiprocessing import set_start_method
from collections import deque
import example_robot_data as robex

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

LOGGER = logging.getLogger(__name__)

# -----------------------
# Meshcat debug helpers
# -----------------------


def make_triad_geom(axis_length=0.08, linewidth=2):
    """
    RGB triad as LineSegments:
      X = red, Y = green, Z = blue
    Compatible with meshcat versions that don't have g.Axes.
    """
    # If your meshcat has Axes, use it
    if hasattr(g, "Axes"):
        # Some versions accept axis_radius, some don't. Keep it simple.
        return g.Axes(axis_length=axis_length)

    # Fallback: LineSegments
    # 6 vertices = 3 segments: O->X, O->Y, O->Z
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
    """
    Empty pointcloud node that we can overwrite in update_debug_visuals.
    Works across meshcat versions.
    """
    P = np.zeros((3, 0), dtype=np.float32)
    C = np.zeros((3, 0), dtype=np.uint8)

    if hasattr(g, "PointCloud"):
        return g.PointCloud(P, C)

    # Older versions: render points
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
    triad_radius=0.003,   # gardé pour compat, pas forcément utilisé en fallback
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

    # Create one triad geometry and reuse it
    # linewidth is a best-effort (WebGL may ignore thickness)
    triad = make_triad_geom(axis_length=triad_length, linewidth=max(1, int(triad_radius * 500)))

    # --- joints triads ---
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        path = f"{root}/joints/{jid:04d}_{jname}"  # <= name visible in Meshcat tree
        vis[path].set_object(triad)
        dbg["joint_entries"].append((jid, path))

    # --- marker frame triads ---
    for mk in marker_names:
        try:
            fid = model.getFrameId(mk)
        except Exception:
            fid = None

        if fid is None or fid < 0 or fid >= len(model.frames):
            dbg["missing_marker_frames"].append(mk)
            continue

        path = f"{root}/marker_frames/{fid:04d}_{mk}"  # <= name visible in Meshcat tree
        vis[path].set_object(triad)
        dbg["marker_entries"].append((fid, path))

    # Empty pointcloud node for model markers
    vis[dbg["model_marker_path"]].set_object(make_empty_pointcloud())

    if dbg["missing_marker_frames"]:
        print("[DEBUG] marker frames missing in model (not registered / not added):")
        print("        ", dbg["missing_marker_frames"])

    return dbg


def update_debug_visuals(vis, model: pin.Model, data: pin.Data, q, dbg):
    """
    Update the transforms of all debug triads and refresh the model marker pointcloud.
    Robust to missing keys (won't crash).
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # --- joints ---
    for jid, path in dbg.get("joint_entries", []):
        vis[path].set_transform(_pin_se3_to_meshcat_tf(data.oMi[jid]))

    # --- marker frames ---
    marker_points = []
    for fid, path in dbg.get("marker_entries", []):
        oMf = data.oMf[fid]
        vis[path].set_transform(_pin_se3_to_meshcat_tf(oMf))
        marker_points.append(oMf.translation)

    # --- model marker pointcloud ---
    if marker_points:
        P = np.stack(marker_points, axis=1)  # (3, N)
        C = np.tile(np.array([[0], [255], [0]], dtype=np.uint8), (1, P.shape[1]))
        vis[dbg.get("model_marker_path", "debug/model_markers")].set_object(g.PointCloud(P, C))

# -----------------------
# Named measured markers (debug)
# -----------------------

def setup_measured_markers(vis: "meshcat.Visualizer", marker_names: Sequence[str], radius: float = 0.010, color: int = 0xff0000):
    """Create one small sphere per measured marker, under markers/measured/<name>."""
    sphere = g.Sphere(radius)
    mat = g.MeshPhongMaterial(color=color, opacity=0.9)
    for name in marker_names:
        vis[f"markers/measured/{name}"].set_object(sphere, mat)


def update_measured_markers(vis: "meshcat.Visualizer", mks_dict: dict):
    """Update transforms for the measured marker spheres."""
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


def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H = settings.height
    mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
    world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)
    
    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        shared_ts_udp,shared_values_udp,lock_udp,cam_event = create_udp_buffer(settings.marker_mocap_names)
        results_queues = create_pipeline_shared_ressources()

        # Create camera processes
        camera_processes = [
            Camera(list(cameras.keys())[i], 
                camera_buffers[i], 
                camera_timestamps[i], 
                camera_locks[i], 
                frame_counters[i], 
                camera_barrier, 
                stop_event,
                cam_event, 
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

        processes = camera_processes + [pipeline]

        # Start processes
        for p in processes:
            p.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
            # Stop processes
            for process in processes:
                process.stop() if hasattr(process, 'stop') else None
                process.join(timeout=2)
    
    else: # offline mode

        # --- 1. INITIALISATION MESHCAT ---
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

        # Init for the rest
        first_sample = True
        p3d_buffer = deque(maxlen=settings.N)

        # Filter
        num_channel = 3*len(settings.marker_names)
        iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=settings.fs
        )
        iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

        while True:
            t0=time.perf_counter()
            frames = src.read()
            if frames is None:
                break

            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

            nlf_out_2d = nlf_out["poses2d"]

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

            p3d = triangulate_points(
                keypoints_list=keypoints_list,
                mtxs=mtxs,
                dists=dists,
                projections=projections,
            )

            p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)

            p3d_in_world=np.array([np.dot(world_R1_cam,point) + world_T1_cam for point in p3d_np])

            if first_sample:
                for k in range(settings.N):
                    p3d_buffer.append(p3d_in_world)  # add the 1st frame 30 times
            else:
                p3d_buffer.append(p3d_in_world) # add the keypoints to the buffer normally
            
            if len(p3d_buffer) == settings.N:
                p3d_buffer_array = np.array(p3d_buffer)

                # Filter keypoints in world to remove noisy artefacts 
                filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array,(settings.N, 3*len(settings.marker_names))))
                filtered_p3d_buffer = np.reshape(filtered_p3d_buffer,(settings.N, len(settings.marker_names), 3))

                augmented_markers=filtered_p3d_buffer[-1]

                # VISUALISATION OF AUGMENTED MARKERS in RED
                colors = np.zeros_like(augmented_markers.T)
                colors[0, :] = 1.0  # R
                colors[1, :] = 0.0  # G
                colors[2, :] = 0.0  # B

                vis_markers.set_object(
                    g.PointCloud(position=augmented_markers.T, color=colors, size=0.02)
                )

                if first_sample:
                    mks_dict = dict(zip(settings.marker_names, augmented_markers))

                    human = robex.human.HumanLoader(height=settings.human_height, weight=settings.human_weight, gender=settings.human_gender).robot
                    human_model = human.model
                    human_collision_model = human.collision_model
                    human_visual_model = human.visual_model

                    #scale the model to data
                    human_model = scale_human_model(human_model, mks_dict, gender=settings.human_gender, subject_height=settings.human_height)
                    human_model= mks_registration(human_model, mks_dict, gender=settings.human_gender, subject_height=settings.human_height)
                    # human_data = pin.Data(human_model)

                    # Init meshcat viewer for human
                    # Visualizers
                    viz_human = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
                    viz_human.initViewer(vis, open=True)
                    
                    # Don't delete the whole Meshcat tree: keep '/markers' etc.
                    try:
                        vis["ref"].delete()
                    except Exception:
                        pass
                    viz_human.loadViewerModel("ref")

                    viz_human.viewer["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
                    viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])  # Same color → flat background

                    # viz_human.display(pin.neutral(human_model))
                    # # show debug frames at neutral configuration
                    # dbg_q0 = pin.neutral(human_model)
                    # # dbg_vis is created a bit later (after background), so we'll update after it's created
                    # # DEBUG: display joint frames + marker frames + model marker positions
                    # dbg_vis = setup_debug_visuals(vis, human_model, settings.marker_names, triad_length=0.08)
                    # update_debug_visuals(vis, human_model, human_data, dbg_q0, dbg_vis)
                    # input()

                    # IK
                    if settings.ik_type == 'sbs':
                        omega = {}
                        for key in settings.keys_to_track_list:
                            omega[key] = 1
                        q = pin.neutral(human_model)
                        ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

                        q = ik_class.solve_ik_sample_casadi()
                        ik_class._q0 = q
                        viz_human.display(q)

                        # Recalibrate briefly the markers translation in joint frames
                        human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,settings.marker_names)
                        human_data=human_model.createData()

                        ik_class = RT_IK(human_model, mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
                        LOGGER.info("[INFO] Model calibration finished, ready to process...")

                    elif settings.ik_type == 'mhe':
                        ik_class = RT_SWIKA(human_model, settings.keys_to_track_list, settings.N, code = settings.ik_code)

                        x_array = np.zeros((human_model.nq+human_model.nv, settings.N))
                        x_array[6,:]=1
                        u_array = np.zeros((human_model.nv, settings.N))
                        deque_lstm_dict = deque(maxlen=settings.N)
                        for k in range(settings.N):
                            deque_lstm_dict.append(mks_dict)

                        array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list]) for d in deque_lstm_dict]).T

                        x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], settings.cost_weights, settings.dt)

                        q = pin.neutral(human_model)
                        q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                        viz_human.display(q)

                        # Recalibrate briefly the markers translation in joint frames
                        human_model=recalibrate_marker_frames_in_joint_space(human_model,q,mks_dict,settings.marker_names)
                        human_data=human_model.createData()

                        ik_class = RT_SWIKA(human_model, settings.keys_to_track_list, settings.N, code = settings.ik_code)
                        LOGGER.info("[INFO] Model calibration finished, ready to process...")
                    else : 
                        raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")

                    first_sample = False

                else: # Init phase finished
                    mks_dict = dict(zip(settings.marker_names, augmented_markers))

                    # IK directly 
                    if settings.ik_type == 'sbs':
                        ik_class._dict_m = mks_dict
                        q = ik_class.solve_ik_sample_quadprog() 
                        ik_class._q0 = q
                        viz_human.display(q)
                    elif settings.ik_type == 'mhe':
                        deque_lstm_dict.append(mks_dict)
                        array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list]) for d in deque_lstm_dict]).T
                        
                        x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:,-1], settings.cost_weights, settings.dt)

                        q = pin.neutral(human_model)
                        q[:] = np.array(x_array[:human_model.nq,-1]).flatten()
                        viz_human.display(q)
                    else : 
                        raise ValueError("Invalid ik type, should be sbs (sample by sample) or mhe (moving horizon estimation)")
            t1=time.perf_counter()
            print(f"Time elapsed for treating one frame = {t1-t0} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)