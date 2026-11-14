#!/usr/bin/env python3
"""Tiled detection against per-view detection on the same frames.

    python3 scripts/python/paper/detection_mode_parity.py --dataset /root/workspace/COMFI \
        --participant 1012 --task Lifting --out results/detection_mode_parity.json

Both estimators run the NLF backend of ``settings.py``; the tiled one uses
``settings.yolo_tiled_imgsz``. Reports, per view, how often the detector found
somebody and the IoU of the tracked boxes, then the differences of the per-view
landmarks and of the joint angles after fusion, filter and IK.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np

from nlf_backend_parity import CAMERAS, FRAMES, Trial


def box_iou(a, b):
    """IoU of two xywh boxes."""
    import torch
    from torchvision.ops import box_convert, box_iou as iou
    a, b = (box_convert(torch.tensor([box], dtype=torch.float64), "xywh", "xyxy") for box in (a, b))
    return float(iou(a, b))


def measure(dataset, participant, task, settings, frames=FRAMES, cameras=CAMERAS):
    """Detection rates, box IoU and output differences of tiled against per-view detection."""
    from rtcosmik.nlf.nlf import extract_views

    trial = Trial(dataset, participant, task, cameras, settings)
    backend = trial.settings.nlf_backend
    modes = {mode: trial.estimator(backend, mode) for mode in ("per_view", "tiled")}
    views = {mode: [] for mode in modes}
    found = {mode: [] for mode in modes}
    tracked = {mode: [] for mode in modes}
    for images in trial.frames(frames):
        for mode, estimator in modes.items():
            out, _, detections, boxes = estimator.estimate_from_frames(images)
            views[mode].append(extract_views(out, len(cameras)))
            found[mode].append([len(d.boxes) > 0 for d in detections])
            tracked[mode].append([b[0].tolist() if len(b) else None for b in boxes])

    iou = [[box_iou(a[v], b[v]) for a, b in zip(tracked["per_view"], tracked["tiled"])
            if a[v] and b[v]] for v in range(len(cameras))]
    layout = modes["tiled"].layout
    report = {
        "participant": participant, "task": task, "cameras": cameras,
        "frames": len(views["per_view"]), "nlf_backend": modes["tiled"].backend,
        "tiled_imgsz": list(layout.canvas), "view_pixels_in_tile": list(layout.tile),
        "detection_rate": {mode: np.mean(found[mode], axis=0).tolist() for mode in modes},
        "box_iou": {"median": [float(np.median(v)) for v in iou],
                    "p1": [float(np.percentile(v, 1)) for v in iou],
                    "min": [float(np.min(v)) for v in iou]},
        **trial.differences(views["per_view"], views["tiled"]),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", help="JSON report")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    text = json.dumps(measure(args.dataset, args.participant, args.task, settings), indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
