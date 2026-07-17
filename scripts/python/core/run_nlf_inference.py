#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import time
from typing import List

import cv2

import numpy as np
import torch
from rtcosmik.nlf.nlf import NLFEstimator, DisplayConsumerNLF, check_yolo_engine
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources

from rtcosmik.utils.videoReader import OfflineVideoSource

from multiprocessing import set_start_method

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir.resolve()}")
    vids = [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in [".mp4"]]
    return vids



def main(args):

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H =settings.height

    if args.online:
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, NUM_CAMERAS)
        check_yolo_engine(NUM_CAMERAS)
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
        if args.videos and len(args.videos) > 0:
            paths = [Path(v) for v in args.videos]
        else:
            paths = list_videos(Path(args.data_dir))
        if len(paths) == 0:
            raise RuntimeError(f"No videos found in {args.data_dir}")
        
        NUM_CAMERAS=len(paths)
        check_yolo_engine(NUM_CAMERAS)
        mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, NUM_CAMERAS)
        
        '''
        To save output videos to check if they are synchronized
        fps=40
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_stream1 = cv2.VideoWriter('sync_check_video1.mp4', fourcc, fps, (W, H))
        out_stream2 = cv2.VideoWriter('sync_check_video2.mp4', fourcc, fps, (W, H))
        '''

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

        cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)

        
        while True:
            frames = src.read()
            if frames is None:
                break
            
            '''
            To save output videos to check if they are synchronized
            out_stream1.write(frames[0])
            out_stream2.write(frames[1])
            '''

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

        src.release()
        
        '''
        To save output videos to check if they are synchronized
        out_stream1.release()
        out_stream2.release()
        '''
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)