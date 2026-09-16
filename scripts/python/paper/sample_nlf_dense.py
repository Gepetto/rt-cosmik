#!/usr/bin/env python3
"""Sample NLF's dense SMPL-X output on the frames the marker-map fit needs.

The FastSAM marker map was learned by ``learn_cosmik_mhr_markers_multisubject.py``
from 20 frames per trial, picked among 100 evenly spaced candidates. Fitting
NLF's markers the same way needs NLF's output on exactly those candidates, and
nothing else, so this runs NLF on camera 0 at all 10475 SMPL-X canonical points
for those frames only and stores them.

Candidate frames are chosen exactly as that script chooses them::

    np.unique(np.linspace(0, n - 1, min(n, 100)).round())

with ``n`` the trial's mocap length, which the video matches frame for frame.

NLF's detector locks onto a box from one frame to the next. Consecutive
candidates here are ~25 frames apart, where a held-over box could crop the
wrong region, so the lock is reset before every sampled frame: each frame gets
a fresh detection of the most confident person, which in these single-subject
trials is the subject.

    python3 scripts/python/paper/sample_nlf_dense.py --out /root/workspace/nlf_dense_samples
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

CANDIDATES = 100
VERTICES = 10475
CAMERA = 0
MARKER_MAP = Path("/root/workspace/COMFI/fastsam/results_multicam/"
                  "cosmik_mhr_marker_map_17subjects_tv8_tv12.json")


def mocap_length(dataset, participant, task):
    path = dataset / "mocap" / "aligned" / participant / task / "markers_trajectories.csv"
    with open(path) as handle:
        return sum(1 for _ in handle) - 1


def candidate_frames(n):
    return np.unique(np.linspace(0, n - 1, min(n, CANDIDATES)).round().astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("/root/workspace/COMFI"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--participants", nargs="*", default=None)
    args = ap.parse_args()

    import sweep as sweep_mod
    from rtcosmik.config_loader import settings
    from rtcosmik.camera.cam_utils import load_camera_parameters
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    mapping = json.loads(MARKER_MAP.read_text())
    tasks = list(mapping["task_directories"].values())
    participants = sorted(set(mapping["subject_ids"].values()))   # the 17 fitted on
    if args.participants:
        participants = [p for p in participants if p in args.participants]

    for participant in participants:
        mtxs, _, _, _, _ = load_camera_parameters(
            args.dataset / "cam_params" / participant, [CAMERA])
        estimator = sweep_mod._nlf_estimator(mtxs, 1, settings,
                                             indices=list(range(VERTICES)))
        for task in tasks:
            dest = args.out / participant / f"{task}.npz"
            if dest.exists():
                continue
            n = mocap_length(args.dataset, participant, task)
            wanted = candidate_frames(n)
            wanted_set = set(wanted.tolist())
            vertices = np.full((len(wanted), VERTICES, 3), np.nan, dtype=np.float32)
            valid = np.zeros(len(wanted), dtype=bool)

            started, slot = time.perf_counter(), 0
            source = OfflineVideoSource(
                paths=[str(args.dataset / "videos" / participant / task / f"camera_{CAMERA}.mp4")],
                size_wh=(settings.width, settings.height), loop=False)
            try:
                frame = -1
                while slot < len(wanted):
                    images = source.read()
                    if images is None:
                        break
                    frame += 1
                    if frame not in wanted_set:
                        continue
                    estimator._locked_boxes_xyxy = [None] * estimator.C
                    estimator._lock_missing_count = [0] * estimator.C
                    out, _, _, _ = estimator.estimate_from_frames(images)
                    pose = extract_views(out, 1).poses3d[0]
                    if pose is not None and len(pose) == VERTICES:
                        vertices[slot] = pose
                        valid[slot] = np.isfinite(pose).all()
                    slot += 1
            finally:
                source.release()

            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, frames=wanted, vertices=vertices, valid=valid, n_frames=n)
            print(f"{participant}/{task}: {int(valid.sum())}/{len(wanted)} valid of {n} frames, "
                  f"{time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
