"""Terminal hotkeys: the SSH-friendly replacement for the X keyboard hook."""
import os
import pty
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.saver.hotkeys import TerminalHotkeys      # noqa: E402


def test_no_tty_is_reported_not_crashed(monkeypatch):
    """A piped or detached stdin must degrade, not raise."""
    monkeypatch.setattr(sys, "stdin", open(os.devnull))
    keys = TerminalHotkeys({"s": lambda: None})
    assert keys.available() is False
    with keys:
        assert keys.active is False


def test_keys_fire_their_actions_on_a_real_terminal(monkeypatch):
    """A pty stands in for the SSH session where pynput cannot work."""
    controller, follower = pty.openpty()
    monkeypatch.setattr(sys, "stdin", os.fdopen(follower, "r"))
    fired = []
    keys = TerminalHotkeys({"s": lambda: fired.append("start"),
                            "q": lambda: fired.append("stop")})
    with keys:
        assert keys.active is True
        os.write(controller, b"sq")
        for _ in range(100):
            if len(fired) >= 2:
                break
            time.sleep(0.01)
    assert fired == ["start", "stop"]


def test_unbound_keys_are_ignored(monkeypatch):
    controller, follower = pty.openpty()
    monkeypatch.setattr(sys, "stdin", os.fdopen(follower, "r"))
    fired = []
    with TerminalHotkeys({"s": lambda: fired.append("start")}):
        os.write(controller, b"xyz s")
        for _ in range(100):
            if fired:
                break
            time.sleep(0.01)
    assert fired == ["start"]


def test_a_failing_handler_does_not_stop_the_listener(monkeypatch):
    controller, follower = pty.openpty()
    monkeypatch.setattr(sys, "stdin", os.fdopen(follower, "r"))
    fired = []
    with TerminalHotkeys({"s": lambda: 1 / 0,
                          "q": lambda: fired.append("stop")}):
        os.write(controller, b"s")
        time.sleep(0.1)
        os.write(controller, b"q")
        for _ in range(100):
            if fired:
                break
            time.sleep(0.01)
    assert fired == ["stop"]
