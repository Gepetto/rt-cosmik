import cv2
import numpy as np
from datetime import datetime

class Camera:
    """
    A class to represent a camera and handle its operations.
    Attributes:
    -----------
    camera_id : int
        The ID of the camera to be used.
    width : int
        The width of the video frames.
    height : int
        The height of the video frames.
    fps : int
        The frames per second of the video.
    fourcc : str
        The four-character code for the video codec.
    _cap : cv2.VideoCapture
        The OpenCV VideoCapture object.
    Methods:
    --------
    __init__(camera_id, width, height, fps, fourcc):
        Initializes the camera with the given parameters and opens it.
    open():
        Opens and configures the camera.
    read_frame():
        Reads a frame from the camera.
    release():
        Releases the camera and destroys all OpenCV windows.
    """
      
    def __init__(self, camera_id, width, height, fps, fourcc):
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._cap = None

        self.open()

    def open(self):
        self._cap = cv2.VideoCapture(self._camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise Exception(f"Camera {self._camera_id} could not be opened.")
        
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read_frame(self):
        if self._cap is None:
            raise Exception("Camera is not opened. Call open() first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ret, frame = self._cap.read()
        if not ret:
            raise Exception(f"Failed to capture frame from camera {self._camera_id}")
        return timestamp, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

def start_camera_process(id_cam, width, height, fps, fourcc, image_buffer, timestamp_buffer, lock, barrier, stopping_event, recording_event):
    """
    Start a camera process to capture frames and store them in a shared memory buffer.
    Args:
        id_cam (int): The ID of the camera to be used.
        width (int): The width of the video frames.
        height (int): The height of the video frames.
        fps (int): The frames per second of the video.
        fourcc (str): The four-character code for the video codec.
        buffer (multiprocessing.Array): The shared memory buffer to store frames.
        lock (multiprocessing.Lock): The lock to synchronize access to the shared buffer.
        barrier (multiprocessing.Barrier): The barrier to synchronize process start.
        stopping_event (multiprocessing.Event): The event to signal process termination.
    """
    barrier.wait() #wait for all process before launching cameras
    camera = Camera(camera_id=id_cam, 
                    width=width, 
                    height=height, 
                    fps=fps, 
                    fourcc=fourcc)

    print(stopping_event)
    print(stopping_event.is_set())
    try:
        while not stopping_event.is_set():
            timestamp, frame = camera.read_frame()

            # Convert timestamp to bytes for shared memory
            ts_bytes = timestamp.encode('utf-8')

            with lock:
                # Update frame buffer
                np_buffer = np.frombuffer(image_buffer.get_obj(), 
                                        dtype=np.uint8).reshape(height, width, 3)
                np_buffer[:] = frame  # Overwrite previous frame
                
                # Update timestamp buffer
                timestamp_np = np.frombuffer(timestamp_buffer.get_obj(), dtype='S23')
                timestamp_np[0] = ts_bytes

    except Exception as e:
        print(f"Camera {id_cam} error: {e}")
    finally:
        camera.release()
        print(f"Camera {id_cam} process terminated")


