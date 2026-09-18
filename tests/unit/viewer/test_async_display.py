"""The display worker must never slow the pipeline down.

That is its whole reason to exist, so these check the drop behaviour rather than
the drawing: a viewer that falls behind has to lose frames, not block.
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.viewer.async_display import AsyncDisplay      # noqa: E402


def test_work_runs_on_another_thread():
    where = {}
    with AsyncDisplay() as display:
        display.submit(lambda: where.setdefault("thread", threading.current_thread().name))
        for _ in range(100):
            if "thread" in where:
                break
            time.sleep(0.01)
    assert where.get("thread") != threading.current_thread().name


def test_drops_instead_of_blocking_when_behind():
    """A flood must not stall the caller; the queue is depth 1 by design."""
    started = time.perf_counter()
    with AsyncDisplay() as display:
        for _ in range(200):
            display.submit(lambda: time.sleep(0.01))
        elapsed = time.perf_counter() - started
        assert display.submitted == 200
        assert display.dropped > 0, "a flood should drop"
    # 200 submissions of 10 ms of work is 2 s if it blocked; it must not.
    assert elapsed < 0.5, f"submitting blocked for {elapsed:.2f}s"


def test_keeps_up_at_pipeline_rate():
    """At the measured rates (34 fps, ~4.7 ms of display) nothing should drop."""
    done = []
    with AsyncDisplay() as display:
        for i in range(30):
            display.submit(lambda i=i: (time.sleep(0.0047), done.append(i)))
            time.sleep(0.029)
        time.sleep(0.2)
    assert display.dropped == 0
    assert len(done) == 30


def test_a_failing_update_does_not_kill_the_run():
    """An offline sweep is about the CSV output, not the preview."""
    after = []
    with AsyncDisplay() as display:
        display.submit(lambda: 1 / 0)
        time.sleep(0.05)
        display.submit(lambda: after.append(True))
        time.sleep(0.1)
    assert after == [True]


def test_disabled_is_a_noop():
    ran = []
    with AsyncDisplay(enabled=False) as display:
        assert display.submit(lambda: ran.append(1)) is False
        time.sleep(0.05)
    assert ran == []
