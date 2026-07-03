#!/usr/bin/env python3
"""Airtight Multi-Stream Benchmarking Script with Dual Output Saving.

Synchronized with production pipeline architectures to maintain absolute data preparation
parity while stripping visualization, writing, and GUI rendering from execution logs.
"""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import time
import logging
import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional

from rtcosmik.nlf.nlf import NLFEstimator
from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import load_camera_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

def pre_load_video(video_path: Path, max_frames=300) -> List[np.ndarray]:
    """Reads entire video into CPU memory as a list of numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    print(f"[I/O] Pre-loading {video_path} into RAM...")
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    print(f"[I/O] Successfully loaded {len(frames)} frames.")
    return frames


def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")
    return [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in [".mp4"]]


def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    W = settings.width
    H = settings.height
    mtxs, _, _, _, _ = load_camera_parameters(settings.cam_calib_path)

    if args.videos and len(args.videos) > 0:
        paths = [Path(v) for v in args.videos]
    else:
        paths = list_videos(Path(args.data_dir))
        
    if len(paths) == 0:
        raise RuntimeError(f"No videos found in {args.data_dir}")
    if len(paths) != 2:
        raise RuntimeError(f"Expected exactly 2 streams, found: {len(paths)}")

    # 1. Pre-load ALL frames using the dynamic resolved system paths
    video1_frames = pre_load_video(paths[0], max_frames=300)
    video2_frames = pre_load_video(paths[1], max_frames=300)

    num_frames = min(len(video1_frames), len(video2_frames))
    if num_frames == 0:
        raise RuntimeError("One or both pre-loaded video streams contain 0 frames.")

    # Initialize estimator using identical settings mappings from run_nlf.py
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

    print(f"\nStarting pipeline processing loop over {num_frames} pre-cached memory frames...")
    
    frame_count = 0
    yolo_history = []
    h2d_pre_history = []
    nlf_history = []
    inference_history = []
    input_overhead_history = []
    total_history = []
    fps_history = []

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10, warmup=0, active=35, repeat=11),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )

    prof.start()

    try:
        # 2. Iterate cleanly through the pre-allocated memory frame slots
        for idx in range(num_frames):
            # ================= TRACKED CORE PIPELINE START =================
            # 1. Clock RAM buffer extraction time instead of disk/decoder I/O
            start_input_overhead = time.perf_counter()
            
            frames = [video1_frames[idx], video2_frames[idx]]
            
            # Auto-resize if frames stored do not match configuration size requirements
            for i in range(len(frames)):
                if frames[i].shape[1] != W or frames[i].shape[0] != H:
                    frames[i] = cv2.resize(frames[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    
            input_overhead_ms = (time.perf_counter() - start_input_overhead) * 1000.0
            
            # 2. Compute engine inference using pure production logic
            nlf_out, infer_timings, yres, boxes = est.estimate_from_frames(frames)
            
            # Map precisely to the keys in your nlf.py dictionary
            yolo_ms = infer_timings.get("yolo_ms", 0.0)
            h2d_pre_ms = infer_timings.get("h2d+pre_ms", 0.0)
            nlf_ms = infer_timings.get("nlf_ms", 0.0)
            pure_inference_ms = infer_timings.get("total_ms", yolo_ms + h2d_pre_ms + nlf_ms)

            # 3. Calculate true math turnaround time (Input Reading + Extracted Model Processing)
            total_ms = input_overhead_ms + pure_inference_ms
            # ================== TRACKED CORE PIPELINE END ==================
            
            current_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            frame_count += 1

            # Store metrics (skipping first 10 warmup frames)
            if frame_count > 10:
                yolo_history.append(yolo_ms)
                h2d_pre_history.append(h2d_pre_ms)
                nlf_history.append(nlf_ms)
                inference_history.append(pure_inference_ms)
                input_overhead_history.append(input_overhead_ms)
                total_history.append(total_ms)
                fps_history.append(current_fps)

            prof.step()

    finally:
        prof.stop()

        # Terminal Performance Summary Report
        if total_history:
            prof.export_chrome_trace("trace.json")
            
            print(f"\n--- Benchmark Statistics (Warmup Frames 1-10 Excluded) ---")
            print(f"Total Processed Frames: {frame_count}")
            print(f"Input Ingestion: mean {np.mean(input_overhead_history):.1f} ms    median {np.median(input_overhead_history):.1f} ms    min {np.min(input_overhead_history):.1f} ms    max {np.max(input_overhead_history):.1f} ms")
            print(f"YOLO Detector:   mean {np.mean(yolo_history):.1f} ms    median {np.median(yolo_history):.1f} ms    min {np.min(yolo_history):.1f} ms    max {np.max(yolo_history):.1f} ms")
            print(f"GPU Pre+H2D:     mean {np.mean(h2d_pre_history):.1f} ms    median {np.median(h2d_pre_history):.1f} ms    min {np.min(h2d_pre_history):.1f} ms    max {np.max(h2d_pre_history):.1f} ms")
            print(f"NLF Localizer:   mean {np.mean(nlf_history):.1f} ms    median {np.median(nlf_history):.1f} ms    min {np.min(nlf_history):.1f} ms    max {np.max(nlf_history):.1f} ms")
            print(f"Net Inference:   mean {np.mean(inference_history):.1f} ms    median {np.median(inference_history):.1f} ms    min {np.min(inference_history):.1f} ms    max {np.max(inference_history):.1f} ms")
            print(f"Total Time:      mean {np.mean(total_history):.1f} ms    median {np.median(total_history):.1f} ms    min {np.min(total_history):.1f} ms    max {np.max(total_history):.1f} ms")
            print(f"Performance:     mean {np.mean(fps_history):.1f} FPS   median {np.median(fps_history):.1f} FPS   min {np.min(fps_history):.1f} FPS   max {np.max(fps_history):.1f} FPS\n\n")

            print("="*37 + " PYTORCH COMPLETE GPU KERNEL PROFILE REPORT " + "="*36)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
            print("="*115 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data", help="Folder containing input videos")
    p.add_argument("--videos", nargs="*", default=None, help="Optional explicit list of input videos")
    p.add_argument("--output-1", type=str, default="benchmark_output_cam0.mp4", help="Filename for stream 1 output")
    p.add_argument("--output-2", type=str, default="benchmark_output_cam1.mp4", help="Filename for stream 2 output")
    p.add_argument("--visualize", action="store_true", help="Enable cv2 visualization window preview")
    args = p.parse_args()
    main(args)