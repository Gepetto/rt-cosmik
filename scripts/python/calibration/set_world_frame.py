#!/usr/bin/env python3
"""Set the world frame by pointing a wand at known positions.

    default    a frame marked on the ground, 3 pointed positions
    --robot    the robot base, 4 pointed positions

Only the first camera gets a world pose; the others are placed from it through
the stereo pairs of calibrate_cameras.py.

    python3 scripts/python/calibration/set_world_frame.py --cameras 0 2 4 6
    python3 scripts/python/calibration/set_world_frame.py --cameras 0 2 --robot --from-images --no-show
"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import glob
import os

import cv2
import numpy as np

from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import (
    chain_relative_poses, intrinsics_path, list_cameras, load_cam_params)
from rtcosmik.calibration.files import save_world_pose
from rtcosmik.calibration.world_frame import (
    fuse_world_anchor, ground_frame, invert_pose, robot_frame, wand_detector,
    wand_pose_in_camera, wand_positions_in_camera)

# With fewer cameras a disagreement cannot be blamed on one of them: the
# reference camera's own measurement is kept.
MIN_CAMERAS_TO_FUSE = 3
CHECK_AXIS_LENGTH = 0.1  # m, axes drawn on the check image
DISAGREEMENT_WARN_DEG = 5.0

# frame -> (construction, pointed positions, description)
FRAMES = {
    "ground": (ground_frame, 3, "a frame marked on the ground"),
    "robot": (robot_frame, 4, "the robot base"),
}


def images_glob(images_root, camera_id):
    return os.path.join(images_root, f"images_world_cam_{camera_id}", "color", "*.png")


def draw_wand_overlay(shown, corners, camera_matrix, dist_coeffs):
    """Draw the wand's axes and its tip, where SPACE would record it."""
    pose = wand_pose_in_camera(corners[0], camera_matrix, dist_coeffs)
    if pose is None:
        return
    tip, rvec, tvec = pose
    cv2.drawFrameAxes(shown, camera_matrix, dist_coeffs, rvec, tvec,
                      settings.wand_marker_size / 2)
    image_points, _ = cv2.projectPoints(tip.reshape(1, 3), np.zeros(3), np.zeros(3),
                                        camera_matrix, dist_coeffs)
    point = image_points.reshape(2)
    if not np.all(np.isfinite(point)):
        return
    tip_xy = tuple(int(v) for v in np.round(point))
    marker_xy = tuple(int(v) for v in np.round(corners[0].reshape(-1, 2).mean(axis=0)))
    cv2.line(shown, marker_xy, tip_xy, (0, 255, 255), 2)
    cv2.drawMarker(shown, tip_xy, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.putText(shown, f"tip {tip[2]:.2f} m", (tip_xy[0] + 24, tip_xy[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def capture(camera_ids, images_root, required_images, intrinsics, show=True):
    """SPACE saves one image per camera, once per pointed position."""
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

    detector = wand_detector()
    print(f"Point the wand at each of the {required_images} positions.")
    print("SPACE saves one image per camera, ESC or q finishes.")
    saved = 0
    try:
        while True:
            frames = [cap.read()[1] for cap in captures]
            if any(frame is None for frame in frames):
                continue

            if show:
                tiles = []
                for frame, camera_id in zip(frames, camera_ids):
                    shown = frame.copy()
                    corners, ids, _ = detector.detectMarkers(
                        cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY))
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(shown, corners, ids)
                        draw_wand_overlay(shown, corners, *intrinsics[camera_id])
                    else:
                        cv2.putText(shown, "no wand", (24, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                    tile = cv2.resize(shown, (640, 480), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile, f"cam {camera_id}  {saved}/{required_images}",
                                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    tiles.append(tile)
                cv2.imshow("world frame", np.hstack(tiles))

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                for frame, folder in zip(frames, folders):
                    cv2.imwrite(os.path.join(folder, f"img_{saved}.png"), frame)
                saved += 1
                print(f"  saved position {saved}/{required_images}")
                if saved == required_images:
                    print("  all positions captured")
            elif key in (27, ord("q")):
                break
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()


def save_world_frame_check(camera_ids, reference, world_R_cam, world_T_cam, relative,
                           intrinsics, images_root, root, show=True):
    """Draw the saved world frame on one image per camera, the others through the stereo pairs."""
    ref_R_world, ref_T_world = invert_pose(world_R_cam, world_T_cam)

    tiles = []
    for camera_id in camera_ids:
        names = sorted(glob.glob(images_glob(images_root, camera_id)))
        image = cv2.imread(names[0], 1) if names else None
        if image is None:
            continue
        # p_cam = R_i p_ref + T_i
        R_i, T_i = relative[camera_id]
        cam_R_world = R_i @ ref_R_world
        cam_T_world = R_i @ ref_T_world + T_i

        K, D = intrinsics[camera_id]
        cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(cam_R_world)[0],
                          cam_T_world.reshape(3, 1), CHECK_AXIS_LENGTH, 4)
        role = "reference" if camera_id == reference else "chained"
        cv2.putText(image, f"cam {camera_id} ({role})", (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.putText(image, os.path.basename(names[0]), (24, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        tiles.append(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA))
    if not tiles:
        return None

    check = np.hstack(tiles)
    path = os.path.join(root, "world_frame_check.png")
    cv2.imwrite(path, check)
    if show:
        cv2.imshow("world frame check", check)
        print("  press any key on the check window to finish")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", action="store_true",
                        help="the robot base as world frame (4 pointed positions) "
                             "instead of a frame on the ground (3)")
    parser.add_argument("--cameras", type=int, nargs="+", default=list(settings.cameras),
                        help="cameras that see the wand; the first is the reference")
    parser.add_argument("--cam-params", default=settings.cam_calib_path,
                        help="calibration directory, holding the intrinsics and stereo pairs")
    parser.add_argument("--images", metavar="DIR",
                        default=os.path.join(settings.output_dir, "calibration"),
                        help="where the images_world_cam_<id>/ folders are")
    parser.add_argument("--from-images", action="store_true",
                        help="skip capture and solve from the images on disk")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open windows; needed without a display")
    args = parser.parse_args()

    cameras, root = args.cameras, args.cam_params
    build_frame, required_images, frame_name = FRAMES["robot" if args.robot else "ground"]

    intrinsics = {}
    for camera_id in cameras:
        path = intrinsics_path(root, camera_id)
        if not os.path.isfile(path):
            raise SystemExit(f"missing intrinsics for camera {camera_id}: {path}\n"
                             f"run scripts/python/calibration/calibrate_cameras.py first")
        intrinsics[camera_id] = load_cam_params(path)

    if not args.from_images:
        capture(cameras, args.images, required_images, intrinsics, show=not args.no_show)

    reference = cameras[0]

    print(f"\nWand measurements ({frame_name} as world frame)")
    measured = {}
    for camera_id in cameras:
        pattern = images_glob(args.images, camera_id)
        count = len(glob.glob(pattern))
        if count != required_images:
            hint = "" if count == 0 else \
                f": these look like images for the {'ground' if args.robot else 'robot'} frame"
            print(f"  camera_{camera_id}: {count} images "
                  f"(need {required_images}), skipped{hint}")
            continue
        try:
            points = wand_positions_in_camera(pattern, *intrinsics[camera_id],
                                              expected=required_images)
        except ValueError as error:
            print(f"  camera_{camera_id}: {error}, skipped")
            continue
        # The frame is built in camera coordinates; the camera in the world is its inverse.
        cam_T_world, cam_R_world = build_frame(points)
        measured[camera_id] = invert_pose(cam_R_world, cam_T_world)
        print(f"  camera_{camera_id}: position {np.round(measured[camera_id][1], 3)} m")

    if reference not in measured:
        raise SystemExit(
            f"the reference camera {reference} has no usable wand measurement")

    relative = chain_relative_poses(root, cameras)
    fused_R, fused_T, spread = fuse_world_anchor(measured, relative, reference)

    if len(measured) >= MIN_CAMERAS_TO_FUSE:
        world_R_cam, world_T_cam = fused_R, fused_T
        print(f"\nFused {len(measured)} measurements into camera_{reference}'s world pose.")
    else:
        world_R_cam, world_T_cam = measured[reference]
        if len(measured) > 1:
            print(f"\nUsing camera_{reference}'s own measurement: fusing needs at "
                  f"least {MIN_CAMERAS_TO_FUSE} cameras.")

    if len(measured) > 1:
        print("  how far each camera's estimate is from the fused one:")
        for camera_id, (angle, distance) in sorted(spread.items()):
            print(f"    camera_{camera_id}: {angle:5.2f} deg, {distance * 1000:6.1f} mm")
        worst = max(angle for angle, _ in spread.values())
        if worst > DISAGREEMENT_WARN_DEG:
            print(f"  warning: {worst:.1f} deg disagreement. At least one camera saw the "
                  f"wand badly; check the pointed positions.")

    path = save_world_pose(world_R_cam, world_T_cam, reference, root)
    print(f"\nWorld pose -> {os.path.relpath(path, root)}")
    print(f"  camera_{reference} is at {np.round(world_T_cam, 3)} m in the world. "
          f"If this is not where the camera is, the pose is wrong.")

    check = save_world_frame_check(cameras, reference, world_R_cam, world_T_cam, relative,
                                   intrinsics, args.images, root, show=not args.no_show)
    if check:
        print(f"\nWorld frame drawn on one image per camera -> "
              f"{os.path.relpath(check, root)}")
        print("  the axes must sit on the pointed frame; x red, y green, z blue")

    print(f"\nWritten under {root}")


if __name__ == "__main__":
    main()
