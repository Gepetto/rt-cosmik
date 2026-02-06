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

import cv2
import numpy as np
import torch
from src.rtcosmik.nlf.nlf import NLFEstimator
from src.rtcosmik.config_loader import settings
from src.rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters
from src.rtcosmik.camera.camera import Camera
from src.rtcosmik.utils.mp_utils import create_camera_shared_ressources

from multiprocessing import set_start_method

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)


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

class OnlineCameraSource:
    """Consumes frames from Camera processes writing into shared buffers."""

    def __init__(self, cam_ids: List[int], frame_shape: Tuple[int, int, int], fps: int, fourcc: str):
        import multiprocessing as mp

        self.cam_ids = cam_ids
        self.num_cams = len(cam_ids)
        self.frame_shape = frame_shape
        self.fps = float(fps)
        self.frame_period = 1.0 / max(1e-6, self.fps)

        self.camera_buffers, self.camera_timestamps, self.camera_locks, self.frame_counters, self.barrier, self.stop_event = (
            create_camera_shared_ressources(self.num_cams, frame_shape)
        )
        self.cam_event = mp.Event()

        # Create numpy views for shared buffers (no copy). We copy under lock at read time.
        self._views = [np.frombuffer(buf, dtype=np.uint8).reshape(frame_shape) for buf in self.camera_buffers]
        self._last_counters = [0] * self.num_cams

        self.processes: List[Camera] = []
        for i, cam_id in enumerate(cam_ids):
            proc = Camera(
                cam_id,
                self.camera_buffers[i],
                self.camera_timestamps[i],
                self.camera_locks[i],
                self.frame_counters[i],
                self.barrier,
                self.stop_event,
                self.cam_event,
                frame_shape=frame_shape,
                cam_fps=fps,
                cam_fourcc=fourcc,
            )
            self.processes.append(proc)

    def start(self):
        for p in self.processes:
            p.start()

    def read(self, timeout_s: float = 1.0) -> Optional[List[np.ndarray]]:
        """Wait until every camera has produced a new frame since last read."""
        t0 = time.perf_counter()
        while not self.stop_event.is_set():
            frames: List[np.ndarray] = []
            new_counters: List[int] = []
            ok_all = True
            for i in range(self.num_cams):
                with self.camera_locks[i]:
                    c = int(self.frame_counters[i].value)
                    if c <= self._last_counters[i]:
                        ok_all = False
                        break
                    frame = self._views[i].copy()  # materialize under lock
                frames.append(frame)
                new_counters.append(c)

            if ok_all and len(frames) == self.num_cams:
                self._last_counters = new_counters
                return frames

            if (time.perf_counter() - t0) > timeout_s:
                return None
            time.sleep(0.001)

        return None

    def stop(self):
        self.stop_event.set()
        for p in self.processes:
            try:
                p.join(timeout=2)
            except Exception:
                pass

def main(args):
    torch.backends.cudnn.benchmark = True

    # Determine size
    W = settings.width
    H =settings.height
    mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path)

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
                cam_event, 
                FRAME_SHAPE, 
                settings.fs, 
                settings.fourcc,)
            for i in range(NUM_CAMERAS)
        ]

        processes = camera_processes

        # Needs to add NLF logic to overlay on the camera images the results

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

            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
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