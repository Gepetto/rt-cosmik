#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import time
from pathlib import Path

import meshcat
import meshcat.geometry as g

import numpy as np
import torch
from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF, extract_views
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.VideoReader import OfflineVideoSource
from rtcosmik.utils.dataset import TRIAL_CLI_EPILOG, add_trial_arguments, resolve_trial
from rtcosmik.model_weights import resolve_detector_engine
from rtcosmik.triangulation.triangulation import reconstruct_3d

from multiprocessing import set_start_method

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

LOGGER = logging.getLogger(__name__)

def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H = settings.height
    if args.online:
        cam_params_path = settings.cam_calib_path
        video_paths = None
    else:
        cam_params_path, video_paths, _, _ = resolve_trial(args)
        if len(video_paths) != len(args.cameras):
            raise ValueError(
                f"{len(video_paths)} videos but {len(args.cameras)} cameras requested; "
                "pass --cameras matching the videos, in the same order"
            )

    mtxs, dists, projections, rotations, translations = load_camera_parameters(
        cam_params_path, args.cameras
    )
    world_R1_cam, world_T1_cam = load_world_transformation(cam_params_path, args.cameras[0])

    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)

        # Create camera processes
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

        paths = video_paths

        NUM_CAMERAS = len(paths)

        src = OfflineVideoSource(paths=paths, size_wh=(W, H), loop=False)

        est = NLFEstimator(
            yolo_path=resolve_detector_engine(settings.yolo_path, len(paths)),
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

            views = extract_views(nlf_out, NUM_CAMERAS)
            p3d = reconstruct_3d(views, projections)
            if len(p3d) == 0:
                continue

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
    p = argparse.ArgumentParser(
        description="Run triangulation live, or offline over one recorded trial.",
        epilog=TRIAL_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--online", action="store_true",
                   help="Capture from live cameras instead of video files")
    add_trial_arguments(p)
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)