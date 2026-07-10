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
    paths: List[Path]
    size_wh: Tuple[int, int]
    queue_size: int = 2  # Keeps 2 frames in flight per stream to maintain speed
    
    # Internal engine tracking
    _procs: List[subprocess.Popen] = field(default_factory=list, init=False)
    # Changed from a single Queue to a List of independent Queues
    _queues: List[Queue] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _threads: List[threading.Thread] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Establish a dedicated isolated queue for EVERY separate video path
        self._queues = [Queue(maxsize=self.queue_size) for _ in self.paths]

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
                '-stream_loop', '-1',      
                '-i', str(p),
                '-vf', filter_graph,
                '-f', 'image2pipe',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-blocksize', str(frame_size), 
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
                    continue 
                
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

            except Exception:
                break

    def read(self) -> Optional[List[np.ndarray]]:
        if not self._procs:
            self._start_pipes()

        assembled_frames = []

        # Force a strict lock-step read across all active channels
        for q in self._queues:
            try:
                # Blocks until THIS specific stream yields its next sequential frame
                frame = q.get(timeout=2.0)
                assembled_frames.append(frame)
                q.task_done()
            except Empty:
                # If any single stream drops out or times out, the whole reader safely halts
                return None

        return assembled_frames if self._running else None

    def release(self):
        """Thread-safe teardown sequence that safely cleans up pipes 
        and filters out annoying OS shutdown artifacts."""
        self._running = False
        
        # processes stop gracefully, or terminate
        for proc in self._procs:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass

        # Join the background reader threads safely
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.1)

        # Drain and close the pipes while filtering out "Broken pipe" spam
        for proc in self._procs:
            try:
                if proc:
                    # If there is data left in stderr, read it before closing
                    if proc.stderr:
                        stderr_output = proc.stderr.read().decode('utf-8', errors='ignore')
                        
                        # Filter out the standard SIGPIPE noise line by line
                        for line in stderr_output.splitlines():
                                print(f"[FFmpeg Error] {line}")
                        
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


# OLD OpenCV implementation kept in case
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
