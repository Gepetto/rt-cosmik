import cv2
import os

class VideoSaver:
    def __init__(self, camera_id, save_dir, fps=40, frame_size=(720, 1080)):  # Updated frame_size
        """Initialize video writer."""
        self._camera_id = camera_id
        self._save_dir = save_dir
        self._fps = fps
        self._frame_size = frame_size  # (width, height)
        
        os.makedirs(self._save_dir, exist_ok=True)
        self._video_filename = os.path.join(self._save_dir, f"camera_{self._camera_id}.mp4")  # .mp4 extension
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Updated codec for MP4
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
