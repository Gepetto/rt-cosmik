from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from queue import Queue, Full, Empty
import threading
import os
from pathlib import Path
import subprocess
import numpy as np
import cv2

@dataclass
class OfflineVideoSource:
    # Sentinel pushed onto a stream's queue when that stream reaches its end.
    _EOF = object()


    paths: List[Path]
    size_wh: Tuple[int, int]
    queue_size: int = 2  # Keeps 2 frames in flight per stream to maintain speed
    # Offline processing needs the streams to end so the caller's loop can
    # terminate; looping forever is only useful for open-ended live previews.
    loop: bool = False

    # Internal engine tracking
    _procs: List[subprocess.Popen] = field(default_factory=list, init=False)
    # Changed from a single Queue to a List of independent Queues
    _queues: List[Queue] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _threads: List[threading.Thread] = field(default_factory=list, init=False)
 
    def __post_init__(self):
        # Establish a dedicated isolated queue for EVERY separate video path
        self._queues = [Queue(maxsize=self.queue_size) for _ in self.paths]
        # Spawn ffmpeg processes now, as a one-time setup cost -- num_cameras
        # and paths are fixed for the run, so there's no reason to defer this
        # to the first read() call. Previously this landed inside the first
        # measured read_times entry as a startup spike.
        try:
            self._start_pipes()
        except Exception:
            # If one camera's ffmpeg process fails to launch partway through
            # the loop, don't leak the ones that DID start -- clean them up
            # before propagating the error.
            self.release()
            raise
 
    def _start_pipes(self):
        self.release()
        self._running = True
        w, h = self.size_wh
        frame_size = w * h * 3
        clean_env = os.environ.copy()
        
        for key in list(clean_env.keys()):
            if "VSCODE" in key:
                clean_env.pop(key)
 
        for stream_idx, p in enumerate(self.paths):
            
            '''
            To save output videos to check if they are synchronized
            filter_graph = (
                f"scale={w}:{h},"
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                "text='FRAME\\: %{n}':x=20:y=20:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.6,"
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                "text='TIME\\: %{pts \\: hms}':x=20:y=60:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.6"
            )
            '''
            filter_graph=f"scale={w}:{h}"
 
            command = [
                'ffmpeg',
                '-loglevel', 'error',
                "-hwaccel", 'auto',
                *(['-stream_loop', '-1'] if self.loop else []),
                '-i', str(p),
                '-vf', filter_graph,
                '-f', 'image2pipe',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-blocksize', str(frame_size), 
                '-threads', '2',
                '-'
            ]
            
            proc = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=frame_size,
                env=clean_env
            )  
            self._procs.append(proc)
 
            # Assign each thread its respective isolated queue destination
            t = threading.Thread(
                target=self._pipe_reader_worker, 
                args=(stream_idx, proc, frame_size), 
                daemon=True
            )
            t.start()
            self._threads.append(t)
 
    def _pipe_reader_worker(self, stream_idx: int, proc: subprocess.Popen, frame_size: int):
        """High-speed background worker tracking an isolated data pipe stream."""
        w, h = self.size_wh
        target_queue = self._queues[stream_idx]
        
        while self._running and proc.poll() is None:
            try:
                frame_buffer = np.empty((h, w, 3), dtype=np.uint8)
                bytes_read = proc.stdout.readinto(frame_buffer)

                if bytes_read == 0 or bytes_read is None:
                    # A zero-length read is EOF, not a hiccup: ffmpeg has closed
                    # the pipe. Spinning here would busy-wait until the process
                    # is reaped, so stop and let the sentinel below signal it.
                    break

                while bytes_read < frame_size and self._running:
                    remaining_view = memoryview(frame_buffer)[bytes_read:]
                    extra_bytes = proc.stdout.readinto(remaining_view)
                    if extra_bytes == 0 or extra_bytes is None:
                        break
                    bytes_read += extra_bytes

                if bytes_read == frame_size and self._running:
                    # Push straight to this stream's dedicated queue channel
                    while self._running:
                        try:
                            target_queue.put(frame_buffer, timeout=0.1)
                            break
                        except Full:
                            continue
                elif self._running:
                    # Trailing partial frame: the stream is truncated, so treat
                    # it as the end rather than emitting a half-filled buffer.
                    break

            except Exception:
                break

        # Sentinel so a waiting read() learns the stream ended immediately,
        # instead of stalling for its full timeout on every remaining call.
        try:
            target_queue.put(self._EOF, timeout=0.5)
        except Full:
            pass
 
    def read(self) -> Optional[List[np.ndarray]]:
        assembled_frames = []

        # Force a strict lock-step read across all active channels
        for q in self._queues:
            try:
                # Blocks until THIS specific stream yields its next sequential frame
                frame = q.get(timeout=2.0)
                q.task_done()
            except Empty:
                # If any single stream drops out or times out, the whole reader safely halts
                return None
            if frame is self._EOF:
                # One stream ended, so there is no complete multi-view frame left.
                return None
            assembled_frames.append(frame)

        return assembled_frames if self._running else None
 
    # Teardown must be explicit: use ``with OfflineVideoSource(...) as src`` or
    # a try/finally. There is deliberately no ``__del__`` fallback, because it
    # cannot work here -- the daemon reader threads are bound methods holding a
    # reference to self, so a leaked source is never collected and ``__del__``
    # would only ever fire on sources that had already been released. Measured:
    # with the source garbage-collected but never released, 2 of 2 ffmpeg
    # processes survived; with release() in a finally, 0 of 2.
    #
    # An unreleased decoder does not exit on its own. Its reader thread blocks
    # putting into a full queue, so ffmpeg blocks writing and never reaches EOF,
    # and it sits holding ~288 MiB of GPU for the life of the process. Four
    # cameras a trial adds up fast: a sweep once accumulated 16 of them, 4.6 GB.

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def release(self):
        """Thread-safe teardown sequence that safely cleans up pipes 
        and filters out annoying OS shutdown artifacts.

        Idempotent: calling it twice, or after __del__ has already run, is a
        no-op rather than an error.
        """
        if not self._procs and not self._threads:
            self._running = False
            return
        self._running = False
        
        # Ask each process to exit gracefully first.
        for proc in self._procs:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
 
        # Actually WAIT for it to exit before touching stderr. proc.stderr.read()
        # below blocks until EOF, which only arrives once the process is dead --
        # with -stream_loop -1, ffmpeg finishes its current frame before honoring
        # SIGTERM, so without this wait/kill step, release() (and therefore the
        # whole program) could hang indefinitely on shutdown, only resolved by
        # repeated Ctrl+C forwarding SIGINT into that blocking read.
        for proc in self._procs:
            try:
                if proc:
                    proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass
 
        # Join the background reader threads safely -- by now each proc's
        # stdout pipe is closed (process reaped above), so any thread still
        # blocked in stdout.readinto() gets EOF and returns promptly.
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.5)
 
        # Drain and close the pipes while filtering out "Broken pipe" spam.
        # Safe to .read() here without blocking indefinitely: every process
        # was already waited-on/killed above, so stderr is already at EOF.
        for proc in self._procs:
            try:
                if proc:
                    if proc.stderr:
                        try:
                            stderr_output = proc.stderr.read()
                            if stderr_output:
                                for line in stderr_output.decode('utf-8', errors='ignore').splitlines():
                                    print(f"[FFmpeg Error] {line}")
                        except Exception:
                            pass
                        proc.stderr.close()
                    
                    if proc.stdout:
                        proc.stdout.close()
            except Exception:
                pass
 
        # Clear memory queues
        for q in self._queues:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except Empty:
                    break
 
        self._procs = []
        self._threads = []
    
def list_videos(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir does not exist: {data_dir}")
    return [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in [".mp4"]]

#OLD OpenCV implementation kept in case
# @dataclass
# class OfflineVideoSource:
#     points_saved=False
#     paths: List[Path]
#     size_wh: Tuple[int, int]

#     def __post_init__(self):
#         self.caps = [cv2.VideoCapture(str(p)) for p in self.paths]
#         for p, cap in zip(self.paths, self.caps):
#             if not cap.isOpened():
#                 raise RuntimeError(f"Could not open video: {p}")

#     def read(self) -> Optional[List[np.ndarray]]:
#         frames: List[np.ndarray] = []
#         for cap in self.caps:
#             ok, frame = cap.read()
#             if not ok:
#                 self.points_saved=True
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 ok, frame = cap.read()
#                 if not ok:
#                     return None
#             W, H = self.size_wh
#             if frame.shape[1] != W or frame.shape[0] != H:
#                 frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
#             frames.append(frame)
#         return frames

#     def release(self):
#         for cap in self.caps:
#             cap.release()