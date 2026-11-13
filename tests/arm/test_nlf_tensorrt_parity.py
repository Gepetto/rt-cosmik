"""NLF on TensorRT against TorchScript, same frames and boxes. Opt in with ``RTCOSMIK_REGRESSION=1``."""
import copy
import os
from pathlib import Path

import pytest

from .conftest import COMFI, PARTICIPANT, TASK

pytestmark = pytest.mark.skipif(os.environ.get("RTCOSMIK_REGRESSION") != "1",
                                reason="set RTCOSMIK_REGRESSION=1 to run")


@pytest.fixture(scope="module")
def report():
    from nlf_backend_parity import measure
    from rtcosmik.config_loader import settings
    if not Path(settings.nlf_engine_path).exists():
        pytest.skip("build the engine with scripts/python/core/export_nlf_engine.py")
    switched = copy.copy(settings)      # the reference must not follow the settings
    switched.nlf_backend, switched.detection_mode = "tensorrt", "tiled"
    return measure(COMFI, PARTICIPANT, TASK, {"engine": settings.nlf_engine_path}, switched,
                   frames=60, noise_floor=False)


@pytest.fixture(scope="module")
def parity(report):
    return report["engine"]


def test_reference_is_torchscript_with_per_view_detection_whatever_the_settings(report):
    assert report["reference"] == {"nlf_backend": "torchscript", "detection_mode": "per_view"}


def test_tensorrt_backend_sees_the_same_landmarks_in_the_image(parity):
    assert parity["pixel"]["median"] < 0.5
    assert parity["pixel"]["max"] < 10


@pytest.mark.xfail(reason="ticket 22: fp16 engine and TorchScript differ by more than 1 mm; "
                          "TorchScript differs as much from itself without optimize_for_inference",
                   strict=False)
def test_tensorrt_backend_matches_torchscript_within_1_mm(parity):
    assert parity["landmark_mm"]["max"] < 1.0
