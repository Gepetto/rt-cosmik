"""Recording control and CSV output, owned by the process that produces the data.

This used to live in ``ViewerProcess``: the viewer held the keyboard listener,
the recording flag and the CSV writer, and received markers and joint angles over
a queue. That coupled three unrelated jobs to the display, and forced the viewer
process to rebuild the human model for itself -- which it did from ``settings``
defaults, so it drew a differently sized person than the one being estimated.

Recording belongs with the data, not with the drawing. The pipeline process has
the calibrated model and the results already, so it writes them directly and the
display becomes a pure consumer that can be dropped when it falls behind.

The recording flag stays a shared ``Value`` because the video writers run as
separate processes and read it.
"""

import logging
from collections import OrderedDict

from rtcosmik.saver.csv_saver import CSVSaver

LOGGER = logging.getLogger(__name__)


class Recorder:
    """Start/stop recording, and write markers and joint angles while on.

    Args:
        settings: the RT-COSMIK settings object. ``SAVE_CSV`` enables the CSV
            output, ``record_hotkeys`` the keyboard toggle, and
            ``record_on_start`` whether recording begins immediately.
        num_cameras: how many per-camera frame counters to record.
        saving_flag: optional shared ``Value`` read by the video writers, so a
            single toggle drives every recorder in the run.
        logger: optional logger. Messages are pre-formatted rather than passed
            printf-style, because the ROS bridge injects an rclpy logger whose
            ``info(message, **kwargs)`` takes no positional format arguments.
    """

    def __init__(self, settings, num_cameras, saving_flag=None, logger=None):
        self.settings = settings
        self.num_cameras = num_cameras
        self.saving_flag = saving_flag
        self.logger = logger or LOGGER

        self.enabled = bool(getattr(settings, "record_on_start", False))
        self._listener = None
        self.hotkeys_active = False
        self._csv = None
        self._counter_names = [f"Frame_{i}" for i in range(num_cameras)]
        self.rows = 0

        if self.saving_flag is not None:
            self.saving_flag.value = self.enabled

    # -- lifecycle --------------------------------------------------------

    def start(self):
        """Open the CSV files and, if configured, listen for the hotkeys."""
        if self.settings.SAVE_CSV:
            self._csv = CSVSaver(
                self.settings.SAVE_DIR,
                markers_header=self._counter_names + list(self.settings.marker_names),
                joint_angles_header=self._counter_names
                + list(self.settings.joint_angles_names),
            )
            self.logger.info(f"[REC] writing CSV to {self.settings.SAVE_DIR}")

        self.hotkeys_active = False
        if getattr(self.settings, "record_hotkeys", True):
            self._start_hotkeys()

        # The toggle may also arrive through the shared flag, from the terminal
        # listener in the parent process, so do not claim it is untoggleable.
        toggleable = self.hotkeys_active or self.saving_flag is not None
        state = ("ON" if self.enabled else
                 "OFF (press 's' to start)" if toggleable else
                 "OFF and NOT TOGGLEABLE -- set settings.record_on_start = True")
        self.logger.info(f"[REC] csv={self.settings.SAVE_CSV} "
                         f"video={self.settings.SAVE_VID}, recording {state}")
        if not (self.settings.SAVE_CSV or self.settings.SAVE_VID):
            self.logger.warning(
                "[REC] nothing will be saved: both SAVE_CSV and SAVE_VID are False")
        return self

    def _start_hotkeys(self):
        try:
            from pynput import keyboard
        except Exception as exc:
            # Headless runs have no input device; recording still works from
            # settings, it just cannot be toggled live.
            self.logger.debug(
                f"[REC] in-process hotkeys unavailable ({exc}); the terminal "
                f"listener in the parent handles this")
            return

        def on_press(key):
            char = getattr(key, "char", None)
            if char == "s":
                self.set_enabled(True)
            elif char == "q":
                self.set_enabled(False)

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self.hotkeys_active = True
        self.logger.info("[REC] press 's' to start recording, 'q' to stop")

    def set_enabled(self, enabled):
        """Turn recording on or off, keeping the shared flag in step."""
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if self.saving_flag is not None:
            self.saving_flag.value = enabled
        self.logger.info(
            f"[REC] recording {'started' if enabled else 'stopped'}")

    def close(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._csv is not None:
            self._csv.close()
            self._csv = None
        if self.rows:
            self.logger.info(f"[REC] {self.rows} rows written")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- per-frame --------------------------------------------------------

    def poll(self):
        """Adopt the shared flag, which the terminal listener may have changed.

        The hotkey listener runs in the parent process -- a spawned child gets
        /dev/null for stdin -- so the flag is how the toggle reaches here.
        """
        if self.saving_flag is None:
            return self.enabled
        wanted = bool(self.saving_flag.value)
        if wanted != self.enabled:
            self.enabled = wanted
            self.logger.info(
                f"[REC] recording {'started' if wanted else 'stopped'}")
        return self.enabled

    def record(self, frame_counters, mks_dict, q):
        """Write one frame, if recording is on and CSV output is configured."""
        self.poll()
        if not (self.enabled and self._csv is not None):
            return False

        counters = OrderedDict(zip(self._counter_names, frame_counters))
        markers = OrderedDict(counters)
        markers.update((name, mks_dict[name]) for name in self.settings.marker_names
                       if name in mks_dict)
        angles = OrderedDict(counters)
        angles.update(zip(self.settings.joint_angles_names,
                          (float(v) for v in q)))

        self._csv.save_markers(markers)
        self._csv.save_joint_angles(angles)
        self.rows += 1
        return True
