"""Single-key commands read from the controlling terminal.

``pynput`` hooks the windowing system, so it needs an X display or direct access
to an input device. Neither exists over SSH, which is how this is normally driven
-- a laptop connected to the machine running the container. It fails with
``failed to acquire X connection``, and recording cannot be toggled at all.

The terminal is available in exactly that situation, so read it instead. This
also removes the X dependency from a headless server run.

Where this must live: the online pipeline starts its children with the ``spawn``
method, and a spawned child gets ``/dev/null`` for stdin. Only the parent process
owns the terminal, so the listener runs there and communicates through the shared
recording flag that the pipeline and the video writers already watch.
"""

import logging
import os
import select
import sys
import threading

LOGGER = logging.getLogger(__name__)


class TerminalHotkeys:
    """Watch stdin for single keypresses and call the bound action.

    Used as a context manager so the terminal is always restored, including on
    an exception or Ctrl-C::

        with TerminalHotkeys({"s": start, "q": stop}):
            ...

    Args:
        bindings: character -> callable, invoked on that keypress.
        logger: optional logger.
    """

    def __init__(self, bindings, logger=None):
        self.bindings = dict(bindings)
        self.logger = logger or LOGGER
        self.active = False
        self._thread = None
        self._stop = threading.Event()
        self._saved = None

    def available(self):
        """Whether stdin is a terminal we can put into character mode."""
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False

    def start(self):
        if not self.available():
            self.logger.warning(
                "[KEY] stdin is not a terminal, so hotkeys are off; "
                "use settings.record_on_start")
            return self
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(fd)
            # cbreak, not raw: keys arrive immediately but Ctrl-C still works.
            tty.setcbreak(fd)
        except Exception as exc:
            self.logger.warning("[KEY] could not set up the terminal (%s)", exc)
            self._saved = None
            return self

        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="rtcosmik-hotkeys")
        self._thread.start()
        self.logger.info("[KEY] %s", ", ".join(
            f"'{k}'" for k in self.bindings))
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                char = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            except Exception:
                break
            action = self.bindings.get(char)
            if action is None:
                continue
            try:
                action()
            except Exception:
                self.logger.exception("[KEY] handler for %r failed", char)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._saved is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
        self.active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False
