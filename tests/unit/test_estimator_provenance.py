"""What recordings and run information say about the NLF backend and the detection mode."""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from rtcosmik import provenance                                      # noqa: E402
from rtcosmik.paper.nlf_views import ViewsRecorder                   # noqa: E402

SIDECAR = {"vertices": [10, 20], "precision": "fp16", "max_crops": 8, "tensorrt": "10.0"}


def settings(folder, backend="torchscript", mode="per_view"):
    (folder / "nlf.engine").write_bytes(b"engine bytes")
    (folder / "nlf.engine.json").write_text(json.dumps(SIDECAR))
    (folder / "det_tiled_736x1280.engine").write_bytes(b"")
    (folder / "det_tiled_736x1280.engine.meta").write_text("1,736x1280")
    return SimpleNamespace(nlf_backend=backend, nlf_engine_path=str(folder / "nlf.engine"),
                           detection_mode=mode, yolo_tiled_imgsz=(736, 1280),
                           yolo_path=str(folder / "det.engine"), nlf_indices=[1, 2])


def test_default_estimator_is_torchscript_with_per_view_detection(tmp_path):
    assert provenance.estimator_identity(settings(tmp_path), 4) == {
        "nlf_backend": "torchscript", "detection_mode": "per_view"}


def test_tensorrt_backend_records_the_engine_its_sidecar_and_tensorrt(tmp_path):
    engine = provenance.estimator_identity(settings(tmp_path, "tensorrt"), 4)["nlf_engine"]
    assert engine["file"] == "nlf.engine"
    assert engine["sha256"] == hashlib.sha256(b"engine bytes").hexdigest()
    assert engine["sidecar"] == SIDECAR
    assert engine["tensorrt_installed"] == provenance._versions()["tensorrt"]


def test_tiled_detection_records_the_image_size_and_the_detector_engine(tmp_path):
    identity = provenance.estimator_identity(settings(tmp_path, mode="tiled"), 4)
    assert identity["detection_mode"] == "tiled"
    assert identity["tiled_imgsz"] == [736, 1280]
    assert identity["detector_engine"] == "det_tiled_736x1280.engine"


def test_a_single_view_is_recorded_as_detected_per_view(tmp_path):
    identity = provenance.estimator_identity(settings(tmp_path, mode="tiled"), 1)
    assert identity == {"nlf_backend": "torchscript", "detection_mode": "per_view"}


def test_versions_include_tensorrt():
    try:
        import tensorrt
        expected = tensorrt.__version__
    except ImportError:
        expected = None
    assert provenance._versions()["tensorrt"] == expected


def test_recording_is_stamped_with_the_estimator_that_made_it(tmp_path):
    identity = provenance.estimator_identity(settings(tmp_path, "tensorrt", "tiled"), 4)
    ViewsRecorder([0, 2], 3, estimator=identity).save(tmp_path / "views.npz")
    assert provenance.recorded_estimator(tmp_path / "views.npz") == identity


def test_recording_without_the_stamp_is_torchscript_with_per_view_detection(tmp_path):
    np.savez_compressed(tmp_path / "views.npz", cameras=np.asarray([0, 2]))
    assert provenance.recorded_estimator(tmp_path / "views.npz") == {
        "nlf_backend": "torchscript", "detection_mode": "per_view"}


def test_run_information_of_a_replay_names_the_recording_s_estimator(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *a: "")
    recorded = {"nlf_backend": "torchscript", "detection_mode": "per_view"}
    info = provenance.collect_provenance(settings(tmp_path, "tensorrt", "tiled"), {"name": "nlf"},
                                         estimator=recorded)
    assert info["estimator"] == recorded
    assert info["settings"]["nlf_backend"] == "torchscript"
    assert info["settings"]["detection_mode"] == "per_view"


def test_run_information_of_a_direct_run_names_the_settings_estimator(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *a: "")
    info = provenance.collect_provenance(settings(tmp_path, "tensorrt", "tiled"),
                                         {"name": "nlf", "cameras": [0, 2, 4, 6]})
    assert info["estimator"]["nlf_backend"] == "tensorrt"
    assert info["estimator"]["detector_engine"] == "det_tiled_736x1280.engine"
    assert info["settings"]["detection_mode"] == "tiled"
