"""Which detector engine each detection mode loads."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.model_weights import detector_arguments, estimator_arguments   # noqa: E402


def engine(folder, name, meta):
    (folder / name).write_bytes(b"")
    (folder / f"{name}.meta").write_text(meta)


def settings(folder, mode, tiled_imgsz=(640, 640)):
    return SimpleNamespace(yolo_path=str(folder / "det.engine"), detection_mode=mode,
                           yolo_tiled_imgsz=tiled_imgsz)


@pytest.fixture
def weights(tmp_path):
    engine(tmp_path, "det_b1.engine", "1,640")
    engine(tmp_path, "det_b4.engine", "4,640")
    return tmp_path


def test_per_view_detection_loads_the_engine_of_the_camera_count(weights):
    args = detector_arguments(settings(weights, "per_view"), 4)
    assert args["yolo_path"] == str(weights / "det_b4.engine")
    assert args["detection"] == "per_view"


def test_tiled_detection_loads_the_batch_1_engine_whatever_the_camera_count(weights):
    args = detector_arguments(settings(weights, "tiled"), 4)
    assert args["yolo_path"] == str(weights / "det_b1.engine")
    assert args["detection"] == "tiled" and args["tiled_imgsz"] == (640, 640)


def test_tiled_detection_at_another_size_loads_the_engine_built_for_it(weights):
    engine(weights, "det_tiled_736x1280.engine", "1,736x1280")
    args = detector_arguments(settings(weights, "tiled", (736, 1280)), 3)
    assert args["yolo_path"] == str(weights / "det_tiled_736x1280.engine")
    assert args["tiled_imgsz"] == (736, 1280)


def test_missing_tiled_engine_says_how_to_build_it(weights):
    with pytest.raises(FileNotFoundError, match="736x1280.*\n.*fetch_models.sh"):
        detector_arguments(settings(weights, "tiled", (736, 1280)), 4)


@pytest.mark.parametrize("meta", [None, "1,544x960"])
def test_tiled_engine_not_recorded_for_its_size_is_refused(weights, meta):
    engine(weights, "det_tiled_736x1280.engine", meta or "")
    if meta is None:
        (weights / "det_tiled_736x1280.engine.meta").unlink()
    with pytest.raises(RuntimeError, match="736x1280.*fetch_models.sh"):
        detector_arguments(settings(weights, "tiled", (736, 1280)), 4)


@pytest.mark.parametrize("meta", [None, "1,640"])
def test_single_camera_engine_is_not_taken_for_a_square_of_another_size(weights, meta):
    (weights / "det_b1.engine.meta").unlink()
    if meta:
        (weights / "det_b1.engine.meta").write_text(meta)
    with pytest.raises(FileNotFoundError, match="320x320"):
        detector_arguments(settings(weights, "tiled", (320, 320)), 4)


def test_estimator_arguments_add_the_nlf_backend_to_the_detector(weights):
    s = settings(weights, "per_view")
    s.nlf_path, s.nlf_backend, s.nlf_engine_path = "nlf.ts", "tensorrt", "nlf.engine"
    args = estimator_arguments(s, 4)
    assert args == {**detector_arguments(s, 4), "nlf_path": "nlf.ts", "backend": "tensorrt",
                    "engine_path": "nlf.engine"}


def test_a_single_view_is_detected_per_view_in_either_mode(weights):
    assert detector_arguments(settings(weights, "tiled"), 1) \
        == detector_arguments(settings(weights, "per_view"), 1)


def test_unknown_detection_mode_is_refused(weights):
    with pytest.raises(ValueError, match="detection_mode"):
        detector_arguments(settings(weights, "mosaic"), 4)
