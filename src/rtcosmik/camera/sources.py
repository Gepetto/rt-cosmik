"""Frame sources backed by ffmpeg, for live cameras and for replaying recordings.

Why not OpenCV. ``cv2.VideoCapture`` gives almost no control over buffering --
``CAP_PROP_BUFFERSIZE`` is advisory and widely ignored by the V4L2 backend -- so
frames queue up in the driver and arrive late. It also cannot record what it
captures without decoding to BGR and re-encoding, which costs CPU and loses
quality for no reason when the camera is already producing MJPEG.

ffmpeg fixes both. ``-fflags nobuffer -flags low_delay`` keeps latency down, and
a single invocation can write two outputs: the camera's own MJPEG stream copied
to disk with no re-encode, and raw BGR frames on stdout for the pipeline. So
recording becomes nearly free rather than a second decode/encode cycle.

The same class replays a recording as if it were a camera. With ``realtime=True``
ffmpeg's ``-re`` paces the file at its native frame rate, which is what makes a
replay a real test: without pacing the pipeline is simply handed frames as fast
as it can take them, which measures throughput but proves nothing about whether
it keeps up with a live stream or drops frames.

The grab/retrieve split mirrors ``cv2.VideoCapture`` so the barrier protocol in
:class:`rtcosmik.camera.camera.Camera` is unchanged: every camera is told to
fetch a frame, all of them wait, and only then is each frame read out.
"""

import logging
import os
import shutil
import subprocess

import numpy as np

LOGGER = logging.getLogger(__name__)


def ffmpeg_available():
    """Whether an ffmpeg binary is on PATH."""
    return shutil.which("ffmpeg") is not None


class FFmpegSource:
    """One video source: a V4L2 device or a file, optionally recorded as it runs.

    Args:
        source: device path (``/dev/video0``) or a path to a video file.
        width, height: frame size to request.
        fps: frame rate to request from a device; ignored for files, which keep
            their own.
        input_format: the device's own format. ``mjpeg`` is what these cameras
            offer and is what makes ``record_path`` free.
        realtime: pace a file at its native rate, so a replay behaves like a
            live stream. Has no meaning for a device, which is already realtime.
        record_path: when set, the *source* stream is copied to this file with
            no re-encode, alongside the frames handed to the pipeline.
        logger: optional logger.
    """

    def __init__(self, source, width=1280, height=720, fps=40,
                 input_format="mjpeg", realtime=False, record_path=None,
                 logger=None):
        self.source = str(source)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.input_format = input_format
        self.realtime = bool(realtime)
        self.record_path = record_path
        self.logger = logger or LOGGER

        self.is_device = self.source.startswith("/dev/")
        self._frame_bytes = self.width * self.height * 3
        self._proc = None
        self._pending = None
        self.frames_read = 0

    # -- lifecycle --------------------------------------------------------

    def open(self):
        if not ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is not on PATH; install it or use the OpenCV source")
        if not self.is_device and not os.path.isfile(self.source):
            raise FileNotFoundError(f"no such recording: {self.source}")
        if self.record_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.record_path)),
                        exist_ok=True)

        self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        self.logger.info("[CAP] %s %s%s", "device" if self.is_device else "replay",
                         self.source,
                         f", recording to {self.record_path}" if self.record_path else "")
        return self

    def command(self):
        """The ffmpeg invocation, as a list. Exposed so it can be asserted on."""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if self.is_device:
            # Low-latency flags belong on a live device only. On a file they make
            # ffmpeg start decoding mid-GOP, so the first frames come out
            # corrupted (measured: mean pixel difference of 45 against a normal
            # decode of the same frame).
            cmd += ["-fflags", "nobuffer", "-flags", "low_delay",
                    "-f", "v4l2", "-input_format", self.input_format,
                    "-video_size", f"{self.width}x{self.height}",
                    "-framerate", str(self.fps)]
        elif self.realtime:
            # Pace the file at its own frame rate. Without this a replay is just
            # a fast file read and cannot show whether the pipeline keeps up.
            cmd += ["-re"]
        cmd += ["-i", self.source]

        if self.record_path:
            # Stream copy: the camera's MJPEG goes to disk untouched, so
            # recording costs no decode and no re-encode.
            cmd += ["-map", "0:v", "-c", "copy", "-y", self.record_path]

        cmd += ["-map", "0:v", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-"]
        return cmd

    def release(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.release()
        return False

    # -- the cv2.VideoCapture-shaped interface ----------------------------

    def grab(self):
        """Pull the next frame's bytes off the pipe. Cheap; decode is retrieve.

        A pipe read returns whatever is available, which for a 2.7 MB frame is
        routinely a partial buffer -- so this loops until the frame is complete
        rather than treating a short read as end of stream.
        """
        if self._proc is None:
            return False
        buf = bytearray(self._frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self._frame_bytes:
            chunk = self._proc.stdout.readinto(view[got:])
            if not chunk:                      # genuine end of stream
                self._pending = None
                return False
            got += chunk
        self._pending = buf
        return True

    def retrieve(self):
        """Return the frame grabbed last, as an (H, W, 3) BGR array."""
        if self._pending is None:
            return False, None
        frame = np.frombuffer(self._pending, dtype=np.uint8).reshape(
            self.height, self.width, 3)
        self._pending = None
        self.frames_read += 1
        return True, frame

    def read(self):
        """grab + retrieve, for callers that do not need the split."""
        if not self.grab():
            return False, None
        return self.retrieve()

    def isOpened(self):  # noqa: N802 - matches cv2.VideoCapture
        return self._proc is not None and self._proc.poll() is None
