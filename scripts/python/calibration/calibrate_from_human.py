#!/usr/bin/env python3
"""Place the cameras from a recording of a person walking, with no checkerboard or wand.

Writes the camera pairs and a world frame on the floor: z up, x along the first
camera's viewing direction, origin under that camera. The intrinsics must exist.

    python3 scripts/python/calibration/calibrate_from_human.py --videos output/walk
"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse

import numpy as np

from rtcosmik.calibration import human
from rtcosmik.calibration.files import save_stereo_pose, save_world_pose
from rtcosmik.camera.cam_utils import intrinsics_path, load_cam_params
from rtcosmik.config_loader import settings


def landmarks(videos, cameras, intrinsics):
    """(F, C, J, 3) landmarks of the recording, in each camera's frame."""
    from rtcosmik.model_weights import estimator_arguments
    from rtcosmik.nlf.nlf import NLFEstimator, extract_views
    from rtcosmik.utils.VideoReader import OfflineVideoSource

    estimator = NLFEstimator(
        **estimator_arguments(settings, len(cameras)), cano_path=settings.cano_path,
        image_size=(settings.width, settings.height), cam_Ks=intrinsics,
        indices=settings.nlf_indices, conf=settings.yolo_conf, imgsz=settings.yolo_imgsz,
        device=settings.device)
    source = OfflineVideoSource(paths=[videos / f"camera_{c}.mp4" for c in cameras],
                                size_wh=(settings.width, settings.height), loop=False)
    poses = []
    try:
        while (frames := source.read()) is not None:
            views = extract_views(estimator.estimate_from_frames(frames)[0], len(cameras))
            frame = np.full((len(cameras), len(settings.marker_names), 3), np.nan)
            for camera, pose in enumerate(views.poses3d):
                if pose is not None:
                    frame[camera] = pose[:len(settings.marker_names)]
            poses.append(frame)
    finally:
        source.release()
    return np.asarray(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True,
                        help="recording holding camera_<id>.mp4 for every camera")
    parser.add_argument("--cameras", type=int, nargs="+", default=list(settings.cameras),
                        help="camera ids; the first is the reference")
    parser.add_argument("--cam-params", default=settings.cam_calib_path,
                        help="calibration directory, holding the intrinsics")
    args = parser.parse_args()

    cameras, root = args.cameras, args.cam_params
    if len(cameras) < 2:
        raise SystemExit("at least two cameras are needed")
    intrinsics = [load_cam_params(intrinsics_path(root, camera)) for camera in cameras]
    poses3d = landmarks(args.videos, cameras, [np.asarray(K, np.float32) for K, _ in intrinsics])
    try:
        result = human.calibrate(poses3d)
    except human.DegenerateRecording as error:
        raise SystemExit(f"cannot calibrate on this recording: {error}")

    for a in range(len(cameras) - 1):
        (R_a, T_a), (R_b, T_b) = result.poses[a], result.poses[a + 1]
        R = R_b @ R_a.T
        T = T_b - R @ T_a
        save_stereo_pose(*intrinsics[a], *intrinsics[a + 1], R, T, 0.0,
                         cameras[a], cameras[a + 1], root)
    for camera, (R, T), residual in zip(cameras[1:], result.poses[1:], result.residual_m[1:]):
        print(f"camera_{camera}: {np.linalg.norm(T):.3f} m from camera_{cameras[0]}, "
              f"views agree to {1e3 * residual:.0f} mm")
    world_R, world_T = human.floor_frame(result.points, settings.marker_names)
    save_world_pose(world_R, world_T, cameras[0], root)
    print(f"camera_{cameras[0]} is {world_T[2]:.2f} m above the floor\nwritten to {root}")


if __name__ == "__main__":
    main()
