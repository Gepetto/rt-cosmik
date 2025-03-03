import cv2
import numpy as np
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Value, Lock

class Camera(Process):
    def __init__(self, 
                 cam_id: int,
                 shared_buffer: mp.Array,
                 timestamp_buffer: mp.Array, # Character array for timestamp
                 lock: Lock,
                 frame_shape: tuple = (1280, 720, 3),
                 cam_fps: int = None,
                 cam_fourcc: str = "MJPG"):
        super().__init__()
        self.cam_id = cam_id
        self.shared_buffer = shared_buffer
        self.timestamp_buffer = timestamp_buffer  # For timestamp string
        self.lock = lock
        self.running = Value('b', True)
        
        # Video capture parameters
        self.frame_shape = frame_shape  # (height, width, channels)
        self.cam_fps = cam_fps
        self.cam_fourcc = cam_fourcc

        # Validate timestamp buffer size (need 26 chars for format)
        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")

    def run(self):
        # Initialize camera once at start
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise Exception(f"Camera {self.cam_id} could not be opened.")
        
        # Set camera properties once if specified
        if self.frame_shape:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[1])
        if self.cam_fps:
            cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        if self.cam_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cam_fourcc))

        # Prepare shared buffer view
        arr = np.frombuffer(self.shared_buffer.get_obj(), dtype=np.uint8)
        frame_buffer = arr.reshape(self.frame_shape)

        # Main capture loop
        while self.running.value:
            ret, frame = cap.read()
            if not ret:
                break  # Exit on failure

            # Generate timestamp
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # Update shared memory
            with self.lock:
                # Update frame buffer
                np.copyto(frame_buffer, frame)
                
                # Update timestamp buffer
                encoded_ts = timestamp_str.encode('utf-8')
                self.timestamp_buffer[:26] = encoded_ts  # Exact 26-byte copy
        
        cap.release()

    def stop(self):
        self.running.value = False