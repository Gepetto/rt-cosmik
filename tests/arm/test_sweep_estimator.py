"""Sweep passes that differ by NLF backend or detection mode stay apart."""
import json
from types import SimpleNamespace

import pytest

import sweep


def settings(backend="torchscript", mode="per_view"):
    return SimpleNamespace(nlf_backend=backend, detection_mode=mode, yolo_tiled_imgsz=(736, 1280),
                           nlf_engine_path="missing.engine", yolo_path="missing.engine")


@pytest.mark.parametrize("backend, mode, cameras, tag", [
    ("torchscript", "per_view", [0, 2, 4, 6], "4cam_nlf"),
    ("tensorrt", "per_view", [0, 2, 4, 6], "4cam_nlf_trt"),
    ("torchscript", "tiled", [0, 2, 4, 6], "4cam_nlf_tiled"),
    ("tensorrt", "tiled", [0, 2, 4, 6], "4cam_nlf_trt_tiled"),
    ("tensorrt", "tiled", [0], "1cam_nlf_trt"),
])
def test_default_tag_names_a_backend_or_detection_mode_that_is_not_the_default(
        backend, mode, cameras, tag):
    assert sweep.default_tag("nlf", cameras, settings(backend, mode)) == tag


def test_default_tag_of_an_arm_without_nlf_ignores_the_estimator_settings():
    assert sweep.default_tag("mmpose", [0, 2], settings("tensorrt", "tiled")) == "2cam_mmpose"


def _run_folder(tmp_path, info):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    return tmp_path


def test_run_folder_of_another_estimator_is_refused(tmp_path):
    made = {"estimator": {"nlf_backend": "torchscript", "detection_mode": "per_view"}}
    run = _run_folder(tmp_path / "run", made)
    sweep.require_same_estimator(run, [0, 2, 4, 6], settings())
    with pytest.raises(RuntimeError, match="torchscript.*tensorrt.*--tag"):
        sweep.require_same_estimator(run, [0, 2, 4, 6], settings("tensorrt"))


def test_run_folder_older_than_the_stamp_is_torchscript_with_per_view_detection(tmp_path):
    run = _run_folder(tmp_path / "run", {"arm": {}})
    sweep.require_same_estimator(run, [0, 2, 4, 6], settings())
    with pytest.raises(RuntimeError, match="--tag"):
        sweep.require_same_estimator(run, [0, 2, 4, 6], settings(mode="tiled"))


def test_run_folder_that_does_not_exist_is_accepted(tmp_path):
    sweep.require_same_estimator(tmp_path / "none", [0, 2], settings("tensorrt", "tiled"))
