"""Run FastSAM3DBody on one complete COMFI video and save its results."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from rtcosmik.pose_estimator import (
    FastSAM3DBodyConfig,
    FastSAM3DBodyEstimator,
    load_opencv_camera_calibration,
)


COMFI_ROOT = Path(os.environ.get("COMFI_ROOT", "/root/workspace/COMFI"))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = COMFI_ROOT / "videos/1012/Lifting/camera_0.mp4"
CALIBRATION_PATH = (
    COMFI_ROOT / "cam_params/1012/intrinsics/camera_0_intrinsics.yaml"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output/fastsam3dbody/comfi-test/1012/Lifting/camera_0"
)


def make_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    array[:] = np.nan
    return array


def mhr70_metadata(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
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


def mhr127_hierarchy(
    estimator: FastSAM3DBodyEstimator,
) -> tuple[np.ndarray, np.ndarray]:
    skeleton = estimator._estimator.model.head_pose.mhr.character_torch.skeleton
    parents = np.asarray(skeleton.joint_parents.detach().cpu(), dtype=np.int32)
    raw_names = list(skeleton.joint_names)
    names = np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in raw_names],
        dtype="<U128",
    )
    assert parents.shape == (127,)
    assert names.shape == (127,)
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


def export_full_video() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    camera_matrix, distortion = load_opencv_camera_calibration(CALIBRATION_PATH)
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    assert capture.isOpened(), VIDEO_PATH
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert total > 0
    frame_ids = np.arange(total, dtype=np.int64)

    log_path = OUTPUT_DIR / "batch_extract.log"
    with log_path.open("w", buffering=1) as log_file:
        print(f"Loading TensorRT FastSAM3DBody; detailed log: {log_path}")
        with redirect_stdout(log_file), redirect_stderr(log_file):
            estimator = FastSAM3DBodyEstimator(FastSAM3DBodyConfig())

        names70, bones70, colors70 = mhr70_metadata(estimator.config.root)
        parents127, names127 = mhr127_hierarchy(estimator)
        np.save(OUTPUT_DIR / "faces.npy", estimator._estimator.faces.astype(np.int32))
        np.save(OUTPUT_DIR / "mhr70_names.npy", names70)
        np.save(OUTPUT_DIR / "mhr70_bones.npy", bones70)
        np.save(OUTPUT_DIR / "joint_parents_127.npy", parents127)
        np.save(OUTPUT_DIR / "joint_names_127.npy", names127)
        np.save(OUTPUT_DIR / "frame_ids.npy", frame_ids)

        arrays: dict[str, Any] = {
            "keypoints70_cam": make_memmap(
                OUTPUT_DIR / "keypoints70_cam.npy", (total, 70, 3)
            ),
            "joints127_cam": make_memmap(
                OUTPUT_DIR / "joints127_cam.npy", (total, 127, 3)
            ),
            "vertices_cam": make_memmap(
                OUTPUT_DIR / "vertices_cam.npy", (total, 18439, 3)
            ),
            "keypoints70_2d": make_memmap(
                OUTPUT_DIR / "keypoints70_2d.npy", (total, 70, 2)
            ),
            "joint_global_rots": make_memmap(
                OUTPUT_DIR / "joint_global_rots.npy", (total, 127, 3, 3)
            ),
            "cam_t": make_memmap(OUTPUT_DIR / "cam_t.npy", (total, 3)),
            "focal_length": make_memmap(
                OUTPUT_DIR / "focal_length.npy", (total,)
            ),
            "bbox": make_memmap(OUTPUT_DIR / "bbox.npy", (total, 4)),
            "reprojection_rmse_px": make_memmap(
                OUTPUT_DIR / "reprojection_rmse_px.npy", (total,)
            ),
        }
        valid = np.zeros(total, dtype=bool)
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        writer = cv2.VideoWriter(
            str(OUTPUT_DIR / "keypoints2d_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        assert writer.isOpened()

        try:
            for frame_index in range(total):
                ok, frame = capture.read()
                assert ok and frame is not None, frame_index
                rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    result = estimator.estimate(rectified, camera_matrix, None)

                if result is None:
                    writer.write(rectified)
                    log_file.write(f"frame {frame_index}: no person\n")
                else:
                    rmse = reprojection_rmse(
                        result["keypoints70_camera"],
                        result["keypoints70_2d"],
                        camera_matrix,
                    )
                    arrays["keypoints70_cam"][frame_index] = result[
                        "keypoints70_camera"
                    ]
                    arrays["joints127_cam"][frame_index] = result["joints127_camera"]
                    arrays["vertices_cam"][frame_index] = result["vertices_camera"]
                    arrays["keypoints70_2d"][frame_index] = result["keypoints70_2d"]
                    arrays["joint_global_rots"][frame_index] = result[
                        "joint_global_rotations"
                    ]
                    arrays["cam_t"][frame_index] = result["camera_translation"]
                    arrays["focal_length"][frame_index] = result["focal_length"]
                    arrays["bbox"][frame_index] = result["bbox"]
                    arrays["reprojection_rmse_px"][frame_index] = rmse
                    valid[frame_index] = True
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
                    log_file.write(
                        f"frame {frame_index}: {result['inference_ms']:.2f} ms, "
                        f"reproj={rmse:.4f}px\n"
                    )

                completed = frame_index + 1
                if completed == 1 or completed % 100 == 0 or completed == total:
                    print(
                        f"[{completed}/{total}] valid={int(valid[:completed].sum())}",
                        flush=True,
                    )
        finally:
            capture.release()
            writer.release()
            for array in arrays.values():
                array.flush()

    np.save(OUTPUT_DIR / "valid.npy", valid)
    errors = np.asarray(arrays["reprojection_rmse_px"])[valid]
    errors = errors[np.isfinite(errors)]
    metadata = {
        "source_video": str(VIDEO_PATH.resolve()),
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frame_count": total,
        "saved_frame_count": total,
        "valid_frame_count": int(valid.sum()),
        "coordinate_frame": "camera",
        "units": "meter",
        "camera_axes": {"x": "right", "y": "down", "z": "forward"},
        "conversion": "p_camera = p_MHR + pred_cam_t",
        "camera_intrinsics": camera_matrix.tolist(),
        "distortion_coeffs": distortion.tolist(),
        "distortion_applied": True,
        "overlay_frames": "undistorted source video frames",
        "keypoints2d_projection": "u=fx*X/Z+cx; v=fy*Y/Z+cy",
        "keypoints2d_source": "keypoints70_camera reprojected with calibrated K",
        "tensorrt": {
            "detector": str(estimator.config.detector_path),
            "backbone": str(estimator.config.backbone_engine_path),
        },
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "faces": list(estimator._estimator.faces.shape),
        "reprojection_rmse_px_mean": (
            float(errors.mean()) if errors.size else None
        ),
        "reprojection_rmse_px_max": float(errors.max()) if errors.size else None,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@pytest.mark.fastsam3dbody
def test_fastsam3dbody_on_complete_comfi_video_and_save_results() -> None:
    missing = [
        str(path)
        for path in (VIDEO_PATH, CALIBRATION_PATH)
        if not path.is_file()
    ]
    if missing:
        pytest.skip("COMFI sample is not mounted: " + ", ".join(missing))

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FastSAM3DBody requires a visible CUDA GPU")

    config = FastSAM3DBodyConfig().resolved()
    required = [
        config.checkpoint_dir / "model.ckpt",
        config.checkpoint_dir / "assets/mhr_model.pt",
        config.detector_path,
        config.backbone_engine_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("FastSAM3DBody assets are missing: " + ", ".join(missing))

    metadata = export_full_video()
    frame_count = metadata["source_frame_count"]
    assert metadata["saved_frame_count"] == frame_count
    assert metadata["valid_frame_count"] >= int(0.9 * frame_count)

    expected_shapes = {
        "bbox": (frame_count, 4),
        "cam_t": (frame_count, 3),
        "focal_length": (frame_count,),
        "frame_ids": (frame_count,),
        "joint_global_rots": (frame_count, 127, 3, 3),
        "joints127_cam": (frame_count, 127, 3),
        "keypoints70_2d": (frame_count, 70, 2),
        "keypoints70_cam": (frame_count, 70, 3),
        "reprojection_rmse_px": (frame_count,),
        "valid": (frame_count,),
        "vertices_cam": (frame_count, 18439, 3),
    }
    for name, shape in expected_shapes.items():
        assert np.load(OUTPUT_DIR / f"{name}.npy", mmap_mode="r").shape == shape

    overlay = cv2.VideoCapture(str(OUTPUT_DIR / "keypoints2d_overlay.mp4"))
    assert overlay.isOpened()
    assert int(overlay.get(cv2.CAP_PROP_FRAME_COUNT)) == frame_count
    overlay.release()
    print(f"Full COMFI results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_fastsam3dbody_on_complete_comfi_video_and_save_results()