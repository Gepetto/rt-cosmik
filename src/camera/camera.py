import cv2
import subprocess
import multiprocessing as mp
import numpy as np

class SingleCamera:
    """Class to handle a single camera."""
    def __init__(self, camera_id=0, width=640, height=480, fps=30, fourcc="MJPG"):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap = None

    def open(self):
        """Open and configure the camera."""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise Exception(f"Camera {self.camera_id} could not be opened.")
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_frame(self):
        """Read a frame from the camera."""
        if self.cap is None:
            raise Exception("Camera is not opened. Call open() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            raise Exception(f"Failed to capture frame from camera {self.camera_id}")
        return frame

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()



