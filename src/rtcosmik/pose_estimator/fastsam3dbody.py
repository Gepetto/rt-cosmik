"""Single-camera Fast SAM 3D Body adapter for RT-COSMIK.

The external project is intentionally kept out of RT-COSMIK's normal Python
environment.  Run ``scripts/bash/setup_fastsam3dbody.sh`` and execute this
adapter with the resulting interpreter.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _default_fastsam_root() -> Path:
    workspace = Path(__file__).resolve().parents[4]
    return workspace / "deps" / "Fast-SAM-3D-Body"


@dataclass(frozen=True)
class FastSAM3DBodyConfig:
    """Paths and inference settings for the optimized single-person path."""

    root: Path = _default_fastsam_root()
    checkpoint_dir: Path | None = None
    detector_path: Path | None = None
    backbone_engine_path: Path | None = None
    detector_imgsz: int = 640
    detector_confidence: float = 0.5
    image_size: int = 512
    require_tensorrt: bool = True
    compile_decoders: bool = True

    def resolved(self) -> "FastSAM3DBodyConfig":
        root = Path(os.environ.get("FASTSAM3DBODY_ROOT", self.root)).resolve()
        checkpoint_dir = Path(
            self.checkpoint_dir
            or root / "checkpoints" / "sam-3d-body-dinov3"
        ).resolve()
        detector_name = (
            "yolo11m-pose.engine" if self.require_tensorrt else "yolo11m-pose.pt"
        )
        detector_path = Path(
            self.detector_path or root / "checkpoints" / "yolo" / detector_name
        ).resolve()
        backbone_engine_path = Path(
            self.backbone_engine_path
            or checkpoint_dir / "backbone_trt" / "backbone_dinov3_fp16.engine"
        ).resolve()
        return FastSAM3DBodyConfig(
            root=root,
            checkpoint_dir=checkpoint_dir,
            detector_path=detector_path,
            backbone_engine_path=backbone_engine_path,
            detector_imgsz=self.detector_imgsz,
            detector_confidence=self.detector_confidence,
            image_size=self.image_size,
            require_tensorrt=self.require_tensorrt,
            compile_decoders=self.compile_decoders,
        )


def load_opencv_camera_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``K`` and ``D`` matrices from an OpenCV YAML calibration file."""

    path = Path(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open camera calibration: {path}")
    try:
        camera_matrix = storage.getNode("K").mat()
        distortion = storage.getNode("D").mat()
    finally:
        storage.release()

    if camera_matrix is None or np.asarray(camera_matrix).shape != (3, 3):
        raise ValueError(f"Calibration {path} has no valid 3x3 K matrix")
    if distortion is None:
        distortion = np.empty(0, dtype=np.float32)
    return (
        np.asarray(camera_matrix, dtype=np.float32),
        np.asarray(distortion, dtype=np.float32).reshape(-1),
    )


class _SinglePersonDetector:
    """Limit detector output before the three-crop body/hand model runs."""

    def __init__(self, detector: Any):
        self._detector = detector
        self.name = detector.name
        self._previous_center: np.ndarray | None = None

    def run_human_detection(self, image: np.ndarray, **kwargs: Any) -> Any:
        result = self._detector.run_human_detection(image, **kwargs)
        if not isinstance(result, dict):
            return result

        boxes = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return result
        centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        if self._previous_center is None:
            areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0), axis=1)
            index = int(np.argmax(areas))
        else:
            index = int(
                np.argmin(np.linalg.norm(centers - self._previous_center, axis=1))
            )
        self._previous_center = centers[index]

        selected = dict(result)
        selected["boxes"] = boxes[index : index + 1]
        if result.get("keypoints") is not None:
            keypoints = np.asarray(result["keypoints"], dtype=np.float32)
            selected["keypoints"] = keypoints[index : index + 1]
        return selected


class FastSAM3DBodyEstimator:
    """Optimized Fast SAM 3D Body inference for one calibrated RGB camera."""

    def __init__(self, config: FastSAM3DBodyConfig | None = None):
        self.config = (config or FastSAM3DBodyConfig()).resolved()
        self._previous_center: np.ndarray | None = None
        self._camera_matrix: np.ndarray | None = None
        self._camera_tensor: Any | None = None
        self._rectification_key: tuple[Any, ...] | None = None
        self._rectification_maps: tuple[np.ndarray, np.ndarray] | None = None
        self._validate_runtime_files()
        self._configure_fast_path()

        if str(self.config.root) not in sys.path:
            sys.path.insert(0, str(self.config.root))

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Fast SAM 3D Body requires CUDA, but this container has no "
                "visible GPU. Start it with NVIDIA GPU passthrough."
            )
        torch.backends.cudnn.benchmark = True

        # These imports must happen after the performance flags are set because
        # the external project reads several of them at module-import time.
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from tools.build_detector import HumanDetector

        checkpoint = self.config.checkpoint_dir / "model.ckpt"
        mhr_model = self.config.checkpoint_dir / "assets" / "mhr_model.pt"
        model, model_config = load_sam_3d_body(
            checkpoint_path=str(checkpoint),
            mhr_path=str(mhr_model),
            device="cuda",
        )
        detector = _SinglePersonDetector(
            HumanDetector(
                name="yolo_pose",
                device="cuda",
                model=str(self.config.detector_path),
            )
        )
        self._torch = torch
        self._estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_config,
            human_detector=detector,
            human_segmentor=None,
            # Calibrated intrinsics are supplied for every frame, making MoGe
            # unnecessary and removing it from the latency-critical path.
            fov_estimator=None,
        )

    def _validate_runtime_files(self) -> None:
        required = [
            self.config.root / "sam_3d_body" / "__init__.py",
            self.config.checkpoint_dir / "model.ckpt",
            self.config.checkpoint_dir / "model_config.yaml",
            self.config.checkpoint_dir / "assets" / "mhr_model.pt",
            self.config.detector_path,
        ]
        if self.config.require_tensorrt:
            required.append(self.config.backbone_engine_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Fast SAM 3D Body is not prepared; missing:\n  - "
                + "\n  - ".join(missing)
                + "\nRun scripts/bash/setup_fastsam3dbody.sh "
                "--download-models --build-tensorrt."
            )

    def _configure_fast_path(self) -> None:
        flags = {
            "IMG_SIZE": str(self.config.image_size),
            "LAYER_DTYPE": "fp32",
            "USE_COMPILE": "1" if self.config.compile_decoders else "0",
            "USE_COMPILE_BACKBONE": "0",
            "DECODER_COMPILE": "1" if self.config.compile_decoders else "0",
            "COMPILE_MODE": "reduce-overhead",
            "COMPILE_WARMUP_BATCH_SIZES": "1",
            "MHR_USE_CUDA_GRAPH": "0",
            "BODY_INTERM_PRED_LAYERS": "0,1,2",
            "HAND_INTERM_PRED_LAYERS": "0,1",
            "MHR_NO_CORRECTIVES": "1",
            "SKIP_KEYPOINT_PROMPT": "1",
            "PARALLEL_DECODERS": "1",
            "GPU_HAND_PREP": "1",
            "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        }
        if self.config.backbone_engine_path.is_file():
            flags.update(
                USE_TRT_BACKBONE="1",
                TRT_BACKBONE_PATH=str(self.config.backbone_engine_path),
            )
        else:
            flags.update(USE_TRT_BACKBONE="0", TRT_BACKBONE_PATH="")
        os.environ.update(flags)

    @staticmethod
    def _center(box: np.ndarray) -> np.ndarray:
        return 0.5 * (box[:2] + box[2:])

    def _select_person(self, outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not outputs:
            return None
        boxes = np.asarray([item["bbox"] for item in outputs], dtype=np.float32)
        centers = np.asarray([self._center(box) for box in boxes])
        if self._previous_center is None:
            areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0), axis=1)
            index = int(np.argmax(areas))
        else:
            index = int(
                np.argmin(np.linalg.norm(centers - self._previous_center, axis=1))
            )
        self._previous_center = centers[index]
        return outputs[index]

    @staticmethod
    def camera_result(person: dict[str, Any], inference_ms: float) -> dict[str, Any]:
        """Convert one external model result into explicit camera-frame arrays."""

        translation = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(3)
        vertices_local = np.asarray(person["pred_vertices"], dtype=np.float32)
        keypoints_local = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
        joints_local = np.asarray(person["pred_joint_coords"], dtype=np.float32)
        return {
            "bbox": np.asarray(person["bbox"], dtype=np.float32).reshape(4),
            "camera_translation": translation,
            "focal_length": float(np.asarray(person["focal_length"]).reshape(())),
            "vertices_local": vertices_local,
            "vertices_camera": vertices_local + translation[None, :],
            "keypoints70_local": keypoints_local,
            "keypoints70_camera": keypoints_local + translation[None, :],
            "keypoints70_2d": np.asarray(
                person["pred_keypoints_2d"], dtype=np.float32
            ),
            "joints127_local": joints_local,
            "joints127_camera": joints_local + translation[None, :],
            "joint_global_rotations": np.asarray(
                person["pred_global_rots"], dtype=np.float32
            ),
            "inference_ms": float(inference_ms),
        }

    def estimate(
        self,
        frame_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Estimate one person and return MHR geometry in camera coordinates."""

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR frame, got {frame_bgr.shape}")
        camera_matrix = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
        distortion = np.asarray(
            distortion if distortion is not None else [], dtype=np.float32
        ).reshape(-1)
        if distortion.size and np.any(np.abs(distortion) > 0):
            rectification_key = (
                frame_bgr.shape[:2],
                camera_matrix.tobytes(),
                distortion.tobytes(),
            )
            if rectification_key != self._rectification_key:
                height, width = frame_bgr.shape[:2]
                self._rectification_maps = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    distortion,
                    None,
                    camera_matrix,
                    (width, height),
                    cv2.CV_32FC1,
                )
                self._rectification_key = rectification_key
            frame_bgr = cv2.remap(
                frame_bgr,
                self._rectification_maps[0],
                self._rectification_maps[1],
                cv2.INTER_LINEAR,
            )

        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self._camera_matrix is None or not np.array_equal(
            camera_matrix, self._camera_matrix
        ):
            self._camera_matrix = camera_matrix.copy()
            self._camera_tensor = (
                self._torch.from_numpy(self._camera_matrix).unsqueeze(0).cuda()
            )
        self._torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = self._estimator.process_one_image(
            frame_rgb,
            cam_int=self._camera_tensor,
            bbox_thr=self.config.detector_confidence,
            detector_imgsz=self.config.detector_imgsz,
            inference_type="full",
            hand_box_source="yolo_pose",
        )
        self._torch.cuda.synchronize()
        inference_ms = 1e3 * (time.perf_counter() - started)
        person = self._select_person(outputs)
        return None if person is None else self.camera_result(person, inference_ms)
