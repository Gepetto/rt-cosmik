"""Meshcat updates off the pipeline's critical path.

Visualisation cost is real: measured at 4.7 ms of a 29.4 ms offline frame, 16%
of the budget, which alone was the difference between 34 and 40 fps on a 40 fps
dataset. It is also pure overhead for a batch run -- nobody is watching a
2819-frame offline sweep.

This replaced a separate viewer *process* fed by a queue. That process rebuilt
the human model from ``settings`` defaults, so it drew a differently sized person
than the one being estimated, and it also owned recording and the keyboard
toggle -- three unrelated jobs coupled to the display.

A thread is enough, and cheaper: it keeps the *calibrated* model in reach
without pickling a pinocchio model or repeating the calibration elsewhere.

Both halves of the work release the GIL -- pinocchio's forward kinematics is C++
and the meshcat update is a websocket write -- so the display genuinely overlaps
the next frame's inference rather than merely being deferred.

The queue holds one item and *drops* rather than blocks. A viewer that cannot
keep up must never slow the pipeline: dropped frames cost a stutter in a preview
nobody is grading, whereas blocking would put the cost straight back where it
was.
"""

import logging
import queue
import threading

LOGGER = logging.getLogger(__name__)


class AsyncDisplay:
    """Run display callables on a background thread, dropping when behind.

    Use as a context manager so the thread is always joined::

        with AsyncDisplay() as display:
            for frame in stream:
                display.submit(lambda q=q: viz.display(q))

    Args:
        enabled: when False every call is a no-op, so a caller can disable
            visualisation without branching at each use site.
        logger: optional logger.
    """

    def __init__(self, enabled=True, logger=None):
        self.enabled = enabled
        self.logger = logger or LOGGER
        self.submitted = 0
        self.dropped = 0
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="rtcosmik-display")
            self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def _run(self):
        while not self._stop.is_set():
            try:
                work = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if work is None:
                break
            try:
                work()
            except Exception:
                # A broken viewer must not take the run down: an offline sweep
                # is about the CSV output, not the preview.
                self.logger.exception("[VIZ] display update failed; continuing")

    def submit(self, work):
        """Queue one display update, dropping it if the viewer is behind."""
        if not self.enabled:
            return False
        self.submitted += 1
        try:
            self._queue.put_nowait(work)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self):
        """Stop the thread, waiting briefly for the last update to land."""
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        if self.submitted:
            self.logger.info(
                "[VIZ] %d display updates, %d dropped (%.0f%%) -- dropping keeps "
                "the viewer off the critical path",
                self.submitted, self.dropped,
                100.0 * self.dropped / max(self.submitted, 1))
