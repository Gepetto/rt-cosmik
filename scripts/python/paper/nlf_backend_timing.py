#!/usr/bin/env python3
"""Per-stage time of the NLF arm with each NLF backend and detection mode, one to four views.

    python3 scripts/python/paper/nlf_backend_timing.py --dataset /root/workspace/COMFI \
        --participant 1012 --task Lifting --out results/nlf_backend_timing.csv

Stages, in ms per frame with the GPU waited for at each boundary: detection,
image upload, crop, NLF, fusion, IK. TorchScript crops inside its own call, so
its crop time is counted in ``nlf_ms``; tiling the views is counted in
``detection_ms``. The tiled detection uses ``settings.yolo_tiled_imgsz``; with
one view it is the per-view detection. Run it on an otherwise idle GPU.
"""
import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np

WARMUP, FRAMES = 40, 400
CAMERAS = [0, 2, 4, 6]
ARMS = [("torchscript", "per_view"), ("tensorrt", "per_view"), ("tensorrt", "tiled")]
STAGES = ["detection_ms", "upload_ms", "crop_ms", "nlf_ms", "fusion_ms", "ik_ms", "total_ms"]


def time_arm(dataset, participant, task, cameras, backend, settings, detection="per_view"):
    """Median and 95th percentile of each stage over ``FRAMES`` frames."""
    import yaml
    from sweep import _nlf_estimator
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    settings = landmark_maps.apply(landmark_maps.select(settings, participant), settings)
    settings = copy.copy(settings)
    settings.nlf_backend, settings.detection_mode = backend, detection
    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = load_world_transformation(cam_dir, cameras[0])
    paths = [root / "videos" / participant / task / f"camera_{c}.mp4" for c in cameras]

    estimator = _nlf_estimator(mtxs, len(cameras), settings)
    estimator.sync_timings = True
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    marker_filter = MarkerFilter(len(settings.marker_names), settings)
    source = OfflineVideoSource(paths=[str(p) for p in paths],
                                size_wh=(settings.width, settings.height), loop=False)
    rows = []
    try:
        for i in range(WARMUP + FRAMES):
            images = source.read()
            if images is None:
                break
            t0 = time.perf_counter()
            out, t, _, _ = estimator.estimate_from_frames(images)
            t1 = time.perf_counter()
            p3d = reconstruct_3d(extract_views(out, len(cameras)), projections)
            if len(p3d) == 0:
                continue
            p3d = np.asarray(p3d) @ np.asarray(world_R).T + np.asarray(world_T)
            mks = dict(zip(settings.marker_names, marker_filter(p3d)))
            t2 = time.perf_counter()
            solver.solve(mks)
            t3 = time.perf_counter()
            if i >= WARMUP:
                rows.append([t["yolo_ms"], t["h2d+pre_ms"], t.get("crop_ms", np.nan), t["nlf_ms"],
                             (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3])
    finally:
        source.release()
    rows = np.asarray(rows)
    row = {"backend": backend, "detection": detection, "views": len(cameras),
           "frames": len(rows)}
    for j, stage in enumerate(STAGES):
        row[stage] = round(float(np.median(rows[:, j])), 2)
        row[stage.replace("_ms", "_p95_ms")] = round(float(np.percentile(rows[:, j], 95)), 2)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    rows = [time_arm(args.dataset, args.participant, args.task, CAMERAS[:n], arm[0], settings, arm[1])
            for n in range(1, len(CAMERAS) + 1)
            for arm in ARMS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["backend"], row["detection"], row["views"], {s: row[s] for s in STAGES})


if __name__ == "__main__":
    main()
