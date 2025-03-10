import cv2
import os
from multiprocessing import Process
import numpy as np
import time 

class VideoSaver:
    def __init__(self, camera_id, save_dir, fps=40, frame_size=(720, 1280)):  # Updated frame_size
        """Initialize video writer."""
        self._camera_id = camera_id
        self._save_dir = save_dir
        self._fps = fps
        self._frame_size = frame_size  # (width, height)
        
        os.makedirs(self._save_dir, exist_ok=True)
        self._video_filename = os.path.join(self._save_dir, f"camera_{self._camera_id}.mp4")  # .mp4 extension
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Updated codec for MP4
        self._video_writer = cv2.VideoWriter(self._video_filename, fourcc, self._fps, self._frame_size)
        if not self._video_writer.isOpened():
            raise RuntimeError("Failed to initialize VideoWriter")

    def write_frame(self, frame):
        """Write a single frame to the video."""
        if (frame.shape[1], frame.shape[0]) != self._frame_size:  # Check (width, height)
            raise ValueError("Frame size does not match the expected dimensions")
        self._video_writer.write(frame)

    def close(self):
        """Release the video writer."""
        if self._video_writer.isOpened():
            self._video_writer.release()

    def __del__(self):
        self.close()

class VideoSaverProcess(Process):
    def __init__(self, camera_id, shared_buffer, lock, frame_counter, frame_shape, save_dir, fps, stop_event):
        super().__init__()
        self.camera_id = camera_id
        self.shared_buffer = shared_buffer
        self.lock = lock
        self.frame_counter = frame_counter
        self.frame_shape = frame_shape  # (height, width, channels)
        self.save_dir = save_dir
        self.fps = fps
        self.stop_event = stop_event
        self.last_frame_count = 0
        self.target_delay = 1.0 / fps  # Time between frames (e.g., 0.04s for 25 FPS)
        
    def run(self):
        vs = VideoSaver(
            camera_id=self.camera_id,
            save_dir=self.save_dir,
            fps=self.fps,
            frame_size=(self.frame_shape[1], self.frame_shape[0])  # (width, height)
        )

        last_time = time.monotonic()
        
        try:
            while not self.stop_event.is_set():
                # Wait until the next frame is due
                while (time.monotonic() - last_time) < self.target_delay:
                    time.sleep(0.001)  # Precision sleep to avoid CPU hogging

                with self.lock:
                    current_count = self.frame_counter.value
                    if current_count != self.last_frame_count:
                        # Get frame from buffer
                        arr = np.frombuffer(self.shared_buffer, dtype=np.uint8)
                        frame = arr.reshape(self.frame_shape).copy()
                        
                        # Write frame
                        vs.write_frame(frame)
                        self.last_frame_count = current_count
                
                last_time = time.monotonic()
        
        finally:
            vs.close()
            print(f"VideoSaverProcess for Camera {self.camera_id} terminated.")