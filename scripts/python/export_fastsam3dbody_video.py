#!/usr/bin/env python3
"""Export Fast SAM 3D Body results and a 2D overlay for one video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rtcosmik.pose_estimator import (
    FastSAM3DBodyConfig,
    FastSAM3DBodyEstimator,
    load_opencv_camera_calibration,
)


def make_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=shape
    )
    array[:] = np.nan
    return array


def mhr70_metadata(root: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sam_3d_body.metadata.mhr70 import pose_info

    keypoints = pose_info["keypoint_info"]
    skeleton = pose_info["skeleton_info"]
    names = [""] * 70
    name_to_id = {}
    for key, info in keypoints.items():
        index = int(key)
        names[index] = info["name"]
        name_to_id[info["name"]] = index

    bones = []
    colors = []
    for _, info in sorted(skeleton.items(), key=lambda item: int(item[0])):
        first, second = info["link"]
        if first in name_to_id and second in name_to_id:
            bones.append((name_to_id[first], name_to_id[second]))
            colors.append(tuple(int(value) for value in info["color"][::-1]))
    return (
        np.asarray(names, dtype="<U64"),
        np.asarray(bones, dtype=np.int32),
        colors,
    )


def mhr127_hierarchy(estimator: FastSAM3DBodyEstimator) -> tuple[np.ndarray, np.ndarray]:
    skeleton = estimator._estimator.model.head_pose.mhr.character_torch.skeleton
    parents = np.asarray(skeleton.joint_parents.detach().cpu(), dtype=np.int32)
    raw_names = list(skeleton.joint_names)
    names = np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in raw_names],
        dtype="<U128",
    )
    if parents.shape != (127,) or names.shape != (127,):
        raise ValueError(
            f"Unexpected MHR hierarchy: parents={parents.shape}, names={names.shape}"
        )
    return parents, names


def draw_overlay(
    frame: np.ndarray,
    points: np.ndarray,
    bones: np.ndarray,
    colors: list[tuple[int, ...]],
    bbox: np.ndarray,
    frame_id: int,
    rmse: float,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for bone_index, (first, second) in enumerate(bones):
        start, end = points[first], points[second]
        if np.isfinite(start).all() and np.isfinite(end).all():
            cv2.line(
                output,
                tuple(np.round(start).astype(int)),
                tuple(np.round(end).astype(int)),
                colors[bone_index],
                2,
                cv2.LINE_AA,
            )
    for point in points:
        if not np.isfinite(point).all():
            continue
        x, y = np.round(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)

    x1, y1, x2, y2 = np.round(bbox).astype(int)
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.putText(
        output,
        f"frame {frame_id} | reproj {rmse:.3f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def reprojection_rmse(
    keypoints_camera: np.ndarray,
    keypoints_2d: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    valid = (
        np.isfinite(keypoints_camera).all(axis=1)
        & np.isfinite(keypoints_2d).all(axis=1)
        & (keypoints_camera[:, 2] > 1e-6)
    )
    if not valid.any():
        return float("nan")
    points = keypoints_camera[valid]
    projected = np.column_stack(
        (
            camera_matrix[0, 0] * points[:, 0] / points[:, 2]
            + camera_matrix[0, 2],
            camera_matrix[1, 1] * points[:, 1] / points[:, 2]
            + camera_matrix[1, 2],
        )
    )
    difference = projected - keypoints_2d[valid]
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=project / "tests/full/data/camera_0.mp4",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=project / "tests/full/config/c1_params_color.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "output/fastsam3dbody/camera_0",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "metadata.json"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    log_file = (args.output_dir / "batch_extract.log").open("w", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(args.start_frame, 0)
    end = total if args.end_frame is None else min(args.end_frame, total)
    frame_ids = np.arange(start, end, args.stride, dtype=np.int64)
    if not len(frame_ids):
        raise ValueError(f"Empty frame range [{start}, {end})")

    camera_matrix, distortion = load_opencv_camera_calibration(args.calibration)
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    estimator = FastSAM3DBodyEstimator(FastSAM3DBodyConfig())
    names70, bones70, colors70 = mhr70_metadata(estimator.config.root)
    parents127, names127 = mhr127_hierarchy(estimator)
    np.save(args.output_dir / "faces.npy", estimator._estimator.faces.astype(np.int32))
    np.save(args.output_dir / "mhr70_names.npy", names70)
    np.save(args.output_dir / "mhr70_bones.npy", bones70)
    np.save(args.output_dir / "joint_parents_127.npy", parents127)
    np.save(args.output_dir / "joint_names_127.npy", names127)
    np.save(args.output_dir / "frame_ids.npy", frame_ids)

    count = len(frame_ids)
    arrays: dict[str, Any] = {
        "keypoints70_cam": make_memmap(
            args.output_dir / "keypoints70_cam.npy", (count, 70, 3)
        ),
        "joints127_cam": make_memmap(
            args.output_dir / "joints127_cam.npy", (count, 127, 3)
        ),
        "vertices_cam": make_memmap(
            args.output_dir / "vertices_cam.npy", (count, 18439, 3)
        ),
        "keypoints70_2d": make_memmap(
            args.output_dir / "keypoints70_2d.npy", (count, 70, 2)
        ),
        "joint_global_rots": make_memmap(
            args.output_dir / "joint_global_rots.npy", (count, 127, 3, 3)
        ),
        "cam_t": make_memmap(args.output_dir / "cam_t.npy", (count, 3)),
        "focal_length": make_memmap(
            args.output_dir / "focal_length.npy", (count,)
        ),
        "bbox": make_memmap(args.output_dir / "bbox.npy", (count, 4)),
        "reprojection_rmse_px": make_memmap(
            args.output_dir / "reprojection_rmse_px.npy", (count,)
        ),
    }
    valid = np.zeros(count, dtype=bool)

    writer = cv2.VideoWriter(
        str(args.output_dir / "keypoints2d_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.stride,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the 2D overlay video")

    log(f"Video: {args.video.resolve()}")
    log(f"Calibration: {args.calibration.resolve()}")
    log(f"Frames: {frame_ids[0]}..{frame_ids[-1]} stride={args.stride}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    output_index = 0
    frame_index = start
    try:
        while frame_index < end and output_index < count:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start) % args.stride:
                frame_index += 1
                continue

            rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            result = estimator.estimate(rectified, camera_matrix, None)
            if result is None:
                writer.write(rectified)
                log(f"[{output_index + 1}/{count}] frame {frame_index}: no person")
            else:
                rmse = reprojection_rmse(
                    result["keypoints70_camera"],
                    result["keypoints70_2d"],
                    camera_matrix,
                )
                arrays["keypoints70_cam"][output_index] = result[
                    "keypoints70_camera"
                ]
                arrays["joints127_cam"][output_index] = result["joints127_camera"]
                arrays["vertices_cam"][output_index] = result["vertices_camera"]
                arrays["keypoints70_2d"][output_index] = result["keypoints70_2d"]
                arrays["joint_global_rots"][output_index] = result[
                    "joint_global_rotations"
                ]
                arrays["cam_t"][output_index] = result["camera_translation"]
                arrays["focal_length"][output_index] = result["focal_length"]
                arrays["bbox"][output_index] = result["bbox"]
                arrays["reprojection_rmse_px"][output_index] = rmse
                valid[output_index] = True
                writer.write(
                    draw_overlay(
                        rectified,
                        result["keypoints70_2d"],
                        bones70,
                        colors70,
                        result["bbox"],
                        frame_index,
                        rmse,
                    )
                )
                log(
                    f"[{output_index + 1}/{count}] frame {frame_index}: "
                    f"{result['inference_ms']:.2f} ms, reproj={rmse:.4f}px"
                )
            output_index += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        for array in arrays.values():
            array.flush()
        log_file.close()

    np.save(args.output_dir / "valid.npy", valid)
    errors = np.asarray(arrays["reprojection_rmse_px"])[valid]
    errors = errors[np.isfinite(errors)]
    metadata = {
        "source_video": str(args.video.resolve()),
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frame_count": total,
        "saved_frame_count": count,
        "valid_frame_count": int(valid.sum()),
        "coordinate_frame": "camera",
        "units": "meter",
        "camera_axes": {"x": "right", "y": "down", "z": "forward"},
        "conversion": "p_camera = p_MHR + pred_cam_t",
        "camera_intrinsics": camera_matrix.tolist(),
        "distortion_coeffs": distortion.tolist(),
        "distortion_applied": True,
        "overlay_frames": "undistorted source video frames",
        "tensorrt": {
            "detector": str(estimator.config.detector_path),
            "backbone": str(estimator.config.backbone_engine_path),
        },
        "shapes": {
            name: list(value.shape) for name, value in arrays.items()
        },
        "faces": list(estimator._estimator.faces.shape),
        "reprojection_rmse_px_mean": (
            float(errors.mean()) if errors.size else None
        ),
        "reprojection_rmse_px_max": float(errors.max()) if errors.size else None,
    }
    marker.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {int(valid.sum())}/{count} valid frames to {args.output_dir}")


if __name__ == "__main__":
    main()
