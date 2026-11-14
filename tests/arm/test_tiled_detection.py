"""Tiled detection against per-view detection, same frames. Opt in with ``RTCOSMIK_REGRESSION=1``."""
import os

import pytest

from .conftest import COMFI, PARTICIPANT, TASK

pytestmark = pytest.mark.skipif(os.environ.get("RTCOSMIK_REGRESSION") != "1",
                                reason="set RTCOSMIK_REGRESSION=1 to run")


@pytest.fixture(scope="module")
def report():
    from detection_mode_parity import measure
    from rtcosmik.config_loader import settings
    return measure(COMFI, PARTICIPANT, TASK, settings, frames=60)


def test_tiled_detection_finds_the_person_in_every_view_as_often(report):
    rates = report["detection_rate"]
    assert all(t >= p - 0.02 for t, p in zip(rates["tiled"], rates["per_view"]))


def test_tiled_boxes_overlap_the_per_view_boxes(report):
    assert min(report["box_iou"]["median"]) > 0.85
    assert min(report["box_iou"]["min"]) > 0.5
