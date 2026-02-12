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
from src.rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from src.rtcosmik.camera.camera import Camera
from src.rtcosmik.utils.mp_utils import create_udp_buffer, create_camera_shared_ressources
from src.rtcosmik.triangulation.triangulation import triangulate_points

from multiprocessing import set_start_method

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

LOGGER = logging.getLogger(__name__)

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
    torch.backends.cudnn.benchmark = True
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

        # Create display consumer
        display = DisplayConsumerNLF(
            settings=settings,
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            stop_event=stop_event,
            mtxs=mtxs,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
            with_triangul=True,
            world_R1_cam=world_R1_cam,
            world_T1_cam=world_T1_cam,
            dists=dists,
            projections=projections,
        )

        processes = camera_processes + [display]

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
        vis_markers2 = vis["markers2"]

        world_M_cam = np.eye(4, dtype=np.float64)
        world_M_cam[:3, :3] = world_R1_cam
        world_M_cam[:3, 3] = world_T1_cam
        vis_markers.set_transform(world_M_cam)
        vis_markers2.set_transform(world_M_cam)

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

        while True:
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

            poses_triangul = torch.from_numpy(p3d).to(dtype=torch.float32)
            poses_cam0=nlf_out['poses3d'][0]/1000

            if nlf_out['poses3d'][0].shape[0] > 0:
                points_all = poses_cam0.view(-1, 3).cpu().numpy().T
                
                colors = np.zeros_like(points_all)
                colors[0, :] = 1.0  # R
                colors[1, :] = 0.0  # G
                colors[2, :] = 0.0  # B

                vis_markers.set_object(
                    g.PointCloud(position=points_all, color=colors, size=0.02)
                )

                points_all2 = poses_triangul.view(-1, 3).cpu().numpy().T
                colors2 = np.zeros_like(points_all2)
                colors2[0, :] = 0.0  # R
                colors2[1, :] = 0.0  # G
                colors2[2, :] = 1.0  # B

                vis_markers2.set_object(
                    g.PointCloud(position=points_all2, color=colors2, size=0.02)
                )

            else:
                vis_markers.delete()
                vis_markers2.delete()

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