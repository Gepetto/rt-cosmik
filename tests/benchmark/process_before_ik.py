#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
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

from rtcosmik.config_loader import settings
from rtcosmik.nlf.nlf import NLFEstimator
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.filtering.iir import IIR
from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation

from multiprocessing import set_start_method
from collections import deque

import logging
import subprocess
import json

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
    frame_counter = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H = settings.height
    mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)
    world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)
    
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

    first_sample = True
    p3d_buffer = deque(maxlen=settings.N)

    # Filter
    num_channel = 3 * len(settings.marker_names)
    iir_filter = IIR(
        num_channel=num_channel,
        sampling_frequency=settings.fs
    )
    iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

    # Storage payload for exporting downstream
    recorded_2d_trajectory = []

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

    print("\nProcessing vision pipeline and recording tracking sequences...")

    while frame_counter < total_frames:
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

        # ----------------------------------------------------
        # EXPORT DATA CAPTURE
        # ----------------------------------------------------
        # Convert numpy coordinate matrices to serializable lists before saving
        serializable_kpts = [kp.tolist() if kp is not None else None for kp in keypoints_list]
        recorded_2d_trajectory.append(serializable_kpts)

        # 3D Spatial Reconstruction
        p3d = triangulate_points(
            keypoints_list=keypoints_list,
            mtxs=mtxs,
            dists=dists,
            projections=projections,
        )

        p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)
        p3d_in_world = np.array([np.dot(world_R1_cam, point) + world_T1_cam for point in p3d_np])

        if first_sample:
            for k in range(settings.N):
                p3d_buffer.append(p3d_in_world)
            first_sample = False
        else:
            p3d_buffer.append(p3d_in_world)
        
        if len(p3d_buffer) == settings.N:
            p3d_buffer_array = np.array(p3d_buffer)

            # High-frequency jitter filtering
            filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array, (settings.N, 3 * len(settings.marker_names))))
            filtered_p3d_buffer = np.reshape(filtered_p3d_buffer, (settings.N, len(settings.marker_names), 3))
            augmented_markers = filtered_p3d_buffer[-1]

            # Render 3D point cloud tracking visualization (Red spheres)
            colors = np.zeros_like(augmented_markers.T)
            colors[0, :] = 1.0  # R
            vis_markers.set_object(
                g.PointCloud(position=augmented_markers.T, color=colors, size=0.02)
            )

        frame_counter += 1

    src.release()

    # ----------------------------------------------------
    # WRITE EXPORTED FILE TO DISK
    # ----------------------------------------------------
    output_data_file = "nlf_2d_keypoints.json"
    with open(output_data_file, "w") as f:
        json.dump(recorded_2d_trajectory, f)
    
    LOGGER.info(f"[INFO] Pipeline processing ended.")
    LOGGER.info(f"[INFO] Successfully exported {len(recorded_2d_trajectory)} frames of 2D keypoints to: '{output_data_file}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)