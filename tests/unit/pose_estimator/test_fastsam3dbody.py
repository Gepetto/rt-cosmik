"""Fast SAM 3D Body smoke test using RT-COSMIK's full-test assets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from rtcosmik.pose_estimator.fastsam3dbody import (
    FastSAM3DBodyConfig,
    FastSAM3DBodyEstimator,
    _SinglePersonDetector,
    load_opencv_camera_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = PROJECT_ROOT / "tests" / "full" / "data" / "camera_0.mp4"
CALIBRATION_PATH = PROJECT_ROOT / "tests" / "full" / "config" / "c1_params_color.yaml"


def test_load_full_camera_calibration() -> None:
    camera_matrix, distortion = load_opencv_camera_calibration(CALIBRATION_PATH)
    assert camera_matrix.shape == (3, 3)
    assert distortion.shape == (5,)
    assert camera_matrix.dtype == np.float32
    assert camera_matrix[0, 0] > 0
    assert camera_matrix[1, 1] > 0
    assert camera_matrix[2, 2] == pytest.approx(1.0)


def test_camera_result_contract() -> None:
    person = {
        "bbox": np.array([10, 20, 100, 200], dtype=np.float32),
        "pred_cam_t": np.array([1, 2, 3], dtype=np.float32),
        "focal_length": np.array(1000, dtype=np.float32),
        "pred_vertices": np.zeros((18439, 3), dtype=np.float32),
        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
        "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
        "pred_joint_coords": np.zeros((127, 3), dtype=np.float32),
        "pred_global_rots": np.zeros((127, 3, 3), dtype=np.float32),
    }
    result = FastSAM3DBodyEstimator.camera_result(person, inference_ms=12.5)
    assert result["vertices_camera"].shape == (18439, 3)
    assert result["keypoints70_camera"].shape == (70, 3)
    assert result["joints127_camera"].shape == (127, 3)
    np.testing.assert_allclose(result["vertices_camera"][0], [1, 2, 3])
    assert result["inference_ms"] == pytest.approx(12.5)
    assert result["focal_length"] == pytest.approx(1000.0)


def test_pytorch_fallback_uses_public_detector_weights() -> None:
    config = FastSAM3DBodyConfig(require_tensorrt=False).resolved()
    assert config.detector_path.name == "yolo11m-pose.pt"


def test_single_person_detector_selects_largest_then_tracks() -> None:
    class Detector:
        name = "yolo_pose"

        def run_human_detection(self, _image: np.ndarray, **_kwargs: object) -> dict:
            return {
                "boxes": np.array(
                    [[0, 0, 10, 10], [20, 20, 50, 60]], dtype=np.float32
                ),
                "keypoints": np.zeros((2, 17, 3), dtype=np.float32),
            }

    detector = _SinglePersonDetector(Detector())
    first = detector.run_human_detection(np.zeros((4, 4, 3), dtype=np.uint8))
    np.testing.assert_allclose(first["boxes"][0], [20, 20, 50, 60])
    assert first["keypoints"].shape == (1, 17, 3)


@pytest.mark.fastsam3dbody
def test_fastsam3dbody_on_full_video(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Fast SAM 3D Body integration test requires a visible CUDA GPU")

    config = FastSAM3DBodyConfig().resolved()
    required = [
        config.checkpoint_dir / "model.ckpt",
        config.checkpoint_dir / "assets" / "mhr_model.pt",
        config.detector_path,
        config.backbone_engine_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("Fast SAM 3D Body assets are missing: " + ", ".join(missing))

    camera_matrix, distortion = load_opencv_camera_calibration(CALIBRATION_PATH)
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    assert capture.isOpened(), VIDEO_PATH
    estimator = FastSAM3DBodyEstimator(config)

    results = []
    frame = None
    # Collect consecutive successful results so the test also exercises the
    # steady-state path after one-time torch.compile work on the first frame.
    for _ in range(20):
        ok, frame = capture.read()
        assert ok and frame is not None
        result = estimator.estimate(frame, camera_matrix, distortion)
        if result is not None:
            results.append(result)
        if len(results) == 6:
            break
    capture.release()

    assert len(results) == 6, "Fewer than six detections in the first 20 frames"
    print(
        "FastSAM inference milliseconds (first + steady-state):",
        [round(result["inference_ms"], 2) for result in results],
    )
    result = results[-1]
    assert result["vertices_camera"].shape == (18439, 3)
    assert result["keypoints70_camera"].shape == (70, 3)
    assert result["keypoints70_2d"].shape == (70, 2)
    assert result["joints127_camera"].shape == (127, 3)
    assert result["joint_global_rotations"].shape == (127, 3, 3)
    for key in (
        "vertices_camera",
        "keypoints70_camera",
        "keypoints70_2d",
        "joints127_camera",
    ):
        assert np.isfinite(result[key]).all(), key

    np.testing.assert_allclose(
        result["vertices_camera"],
        result["vertices_local"] + result["camera_translation"][None, :],
        atol=1e-6,
    )
    np.savez_compressed(
        tmp_path / "fastsam3dbody_smoke_result.npz",
        **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
    )
