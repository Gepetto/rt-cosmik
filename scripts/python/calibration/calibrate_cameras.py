#!/usr/bin/env python3
"""Calibrate the cameras with a checkerboard: intrinsics, then each adjacent pair.

    python3 scripts/python/calibration/calibrate_cameras.py --cameras 0 2 4 6
    python3 scripts/python/calibration/calibrate_cameras.py --cameras 0 2 --from-images --no-show
"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import os

import cv2
import numpy as np

from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import intrinsics_path, list_cameras, load_cam_params
from rtcosmik.calibration.checkerboard import (
    calibrate_intrinsics, calibrate_stereo_pair, count_shared_checkerboard_views)
from rtcosmik.calibration.files import (
    camera_hardware_info, save_camera_manifest, save_intrinsics, save_stereo_pose)

# A pair with fewer shared views is not solved.
MIN_SHARED_VIEWS = 6


def images_glob(images_root, camera_id):
    return os.path.join(images_root, f"images_calib_cam_{camera_id}", "color", "*.png")


def capture(camera_ids, images_root):
    """Show the live views; SPACE saves one image per camera."""
    available = list(list_cameras())
    missing = [c for c in camera_ids if c not in available]
    if missing:
        raise RuntimeError(f"cameras {missing} not found; v4l2 lists {available}")

    captures = [cv2.VideoCapture(index, cv2.CAP_V4L2) for index in camera_ids]
    for cap in captures:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        cap.set(cv2.CAP_PROP_FPS, settings.fs)

    folders = [os.path.dirname(images_glob(images_root, c)) for c in camera_ids]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("SPACE saves one image per camera, ESC or q finishes.")
    saved = 0
    try:
        while True:
            frames = [cap.read()[1] for cap in captures]
            if any(frame is None for frame in frames):
                continue
            tiles = [cv2.resize(f, (640, 480), interpolation=cv2.INTER_NEAREST)
                     for f in frames]
            for tile, camera_id in zip(tiles, camera_ids):
                cv2.putText(tile, f"cam {camera_id}", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("calibration", np.hstack(tiles))

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                for frame, folder in zip(frames, folders):
                    cv2.imwrite(os.path.join(folder, f"img_{saved}.png"), frame)
                saved += 1
                print(f"  saved image set {saved}")
            elif key in (27, ord("q")):
                break
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


def parse_labels(items):
    labels = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--labels expects ID=NAME, got {item!r}")
        key, value = item.split("=", 1)
        labels[int(key)] = value
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="+", default=list(settings.cameras),
                        help="camera ids, in rig order; adjacent ones are paired")
    parser.add_argument("--cam-params", default=settings.cam_calib_path,
                        help="calibration directory to write")
    parser.add_argument("--images", metavar="DIR",
                        default=os.path.join(settings.output_dir, "calibration"),
                        help="where the images_calib_cam_<id>/ folders are")
    parser.add_argument("--from-images", action="store_true",
                        help="skip capture and calibrate from the images on disk")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open detection windows; needed without a display")
    parser.add_argument("--labels", nargs="+", default=None, metavar="ID=NAME",
                        help="names for the cameras, e.g. 0=front_left 2=front_right")
    args = parser.parse_args()

    cameras, root = args.cameras, args.cam_params
    if len(cameras) < 2:
        raise SystemExit("at least two cameras are needed to relate them")
    labels = parse_labels(args.labels)

    if not args.from_images:
        capture(cameras, args.images)

    print("\nIntrinsics")
    for camera_id in cameras:
        pattern = images_glob(args.images, camera_id)
        if not os.path.isdir(os.path.dirname(pattern)):
            raise SystemExit(f"no images for camera {camera_id}: {os.path.dirname(pattern)}")
        reproj, mtx, dist = calibrate_intrinsics(pattern, show=not args.no_show)
        path = save_intrinsics(mtx, dist, reproj, camera_id, root)
        print(f"  camera_{camera_id}: reprojection {float(reproj):.4f} px -> "
              f"{os.path.relpath(path, root)}")

    print("\nStereo pairs")
    for cam_a, cam_b in zip(cameras[:-1], cameras[1:]):
        pattern_a, pattern_b = images_glob(args.images, cam_a), images_glob(args.images, cam_b)
        shared, total = count_shared_checkerboard_views(pattern_a, pattern_b)
        if shared < MIN_SHARED_VIEWS:
            raise SystemExit(
                f"  camera_{cam_a} -> camera_{cam_b}: only {shared} of {total} shots "
                f"show the checkerboard to both cameras (need {MIN_SHARED_VIEWS}).\n"
                f"  Capture more shots with the board where the two views overlap.")
        mtx_a, dist_a = load_cam_params(intrinsics_path(root, cam_a))
        mtx_b, dist_b = load_cam_params(intrinsics_path(root, cam_b))
        rmse, R, T = calibrate_stereo_pair(mtx_a, dist_a, mtx_b, dist_b,
                                           pattern_a, pattern_b, show=not args.no_show)
        path = save_stereo_pose(mtx_a, dist_a, mtx_b, dist_b, R, T, rmse, cam_a, cam_b, root)
        print(f"  camera_{cam_a} -> camera_{cam_b}: {shared}/{total} shared views, "
              f"rmse {float(rmse):.4f}, baseline {np.linalg.norm(T):.4f} m "
              f"-> {os.path.relpath(path, root)}")

    manifest = save_camera_manifest(root, cameras, labels)
    print(f"\nRecorded the rig in {os.path.relpath(manifest, root)}:")
    for camera_id in cameras:
        info = camera_hardware_info(camera_id)
        print(f"  {labels.get(camera_id, f'camera_{camera_id}')}: "
              f"{info.get('model', 'unknown')} on port {info.get('bus_info', 'unknown')}")

    print(f"\nWritten under {root}")
    print("Set the world frame next: scripts/python/calibration/set_world_frame.py")


if __name__ == "__main__":
    main()
