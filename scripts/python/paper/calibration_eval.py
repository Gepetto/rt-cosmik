#!/usr/bin/env python3
"""Human-pose calibration on COMFI against the dataset calibration.

Stages, each resumable, all under ``--out``:

    extract    NLF over each participant's walking recordings -> views/
    calibrate  extrinsics from the circular walk -> cam_params/, geometry.csv
    replay     the six evaluation tasks under that calibration -> runs/, results/
    perturb    the same tasks under perturbed dataset extrinsics -> results/
    report     calibration.md

    python3 scripts/python/paper/calibration_eval.py extract --out /root/workspace/oof/t15
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
CAMERAS = [0, 2, 4, 6]
#: The calibration recording, and a straight walk the calibration must refuse.
WALKS = ("CircularWalking", "StraightWalking")


def extract(out):
    """Per-camera NLF output of every walking recording, with the participant's fold map."""
    import sweep
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters
    from rtcosmik.config_loader import settings as base
    from rtcosmik.folds import PARTICIPANTS
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.paper.nlf_views import ViewsRecorder
    from rtcosmik.provenance import estimator_identity, require_clean
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    require_clean()
    for participant in PARTICIPANTS:
        landmark_map = landmark_maps.select(base, participant)
        landmark_maps.require_out_of_fold(landmark_map, participant, base)
        settings = landmark_maps.apply(landmark_map, base)
        mtxs = load_camera_parameters(DATASET / "cam_params" / participant, CAMERAS)[0]
        estimator = sweep._nlf_estimator(mtxs, len(CAMERAS), settings)
        for task in WALKS:
            path = out / "views" / participant / f"{task}.npz"
            if path.exists():
                continue
            videos = [DATASET / "videos" / participant / task / f"camera_{c}.mp4" for c in CAMERAS]
            source = OfflineVideoSource(paths=[str(v) for v in videos],
                                        size_wh=(settings.width, settings.height), loop=False)
            recorder = ViewsRecorder(CAMERAS, len(settings.marker_names), landmark_map,
                                     estimator_identity(settings, len(CAMERAS)))
            try:
                while (frames := source.read()) is not None:
                    output = estimator.estimate_from_frames(frames)[0]
                    recorder.append(extract_views(output, len(CAMERAS)))
            finally:
                source.release()
            recorder.save(path)
            print(f"{participant}/{task}: {len(recorder.keypoints)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["extract"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"extract": extract}[args.stage](args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
