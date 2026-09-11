#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.VideoReader import OfflineVideoSource
from rtcosmik.utils.dataset import TRIAL_CLI_EPILOG, add_trial_arguments, resolve_trial
from rtcosmik.model_weights import resolve_detector_engine

from multiprocessing import set_start_method

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H =settings.height
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
        paths = video_paths

        src = OfflineVideoSource(paths=paths, size_wh=(W, H), loop=False)
        try:


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

            cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)

            while True:
                frames = src.read()
                if frames is None:
                    break

                nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

                print(f"Timings to perform inference = {infer_ms}")

                vis_frames = est.visualize_frames(
                    frames,
                    nlf_out,
                    boxes=boxes,
                    draw_boxes=True,
                    put_text=True,
                    text_prefix="cam",
                )
                vis = np.hstack(vis_frames)

                cv2.imshow("Visualization", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
        finally:
            # Explicit teardown: an unreleased decoder never exits on
            # its own, it blocks on a full pipe holding GPU memory.
            src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run NLF inference live, or offline over one recorded trial.",
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