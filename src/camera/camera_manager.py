import numpy as np
import multiprocessing as mp
from camera.camera import start_camera_process
from camera.cam_utils import list_cameras
from datetime import datetime

class CameraManager:
    """
    Manages multiple camera processes.
    Attributes:
        _width (int): The width of the camera frames.
        _height (int): The height of the camera frames.
        _fps (int): The frames per second for the camera.
        _fourcc (str): The four-character code for the video codec.
        _cameras (list): A list of dictionaries containing shared resources for each camera.
        _processes (list): A list of multiprocessing.Process objects for each camera.
        _barrier (multiprocessing.Barrier): A barrier to synchronize camera processes.
        _stop_event (multiprocessing.Event): An event to signal stopping of camera processes.
    Methods:
        __init__(width, height, fps, fourcc):
            Initializes the CameraManager with the given parameters and initializes cameras.
        initialize_cameras():
            Discovers and initializes available cameras, setting up shared resources for each.
        get_shared_resources():
            Returns all shared resources for external processes.
    """

    def __init__(self, width, height, fps, fourcc):
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._cameras = []
        self._processes = []
        self._barrier = None
        self._stop_event = mp.Event()

        self.initialize_cameras()

    def initialize_cameras(self):
        """Discover and initialize available cameras"""
        detected_cameras = list_cameras()
        
        self._barrier = mp.Barrier(len(detected_cameras))
        self._cameras = []
        
        for cam_id in detected_cameras.keys():
            # Create shared resources for each camera
            camera_data = {
                'id': cam_id,
                'frame_buffer': mp.Array('B', self._width * self._height * 3),
                'timestamp_buffer': mp.Array('c', 23),
                'lock': mp.Lock()
            }
            self._cameras.append(camera_data)

    def get_shared_resources(self):
        """Return all shared resources for external processes"""
        return {
            'cameras': [
                {
                    'id': cam['id'],
                    'frame_buffer': cam['frame_buffer'],
                    'timestamp_buffer': cam['timestamp_buffer'],
                    'lock': cam['lock'],
                    'shape': (self._height, self._width, 3),
                    'dtype': np.uint8
                } for cam in self._cameras
            ],
            'stop_event': self._stop_event
        }

class CameraBufferReader:
    """
    A class to read frames from multiple cameras atomically.
    Attributes:
    -----------
    _resources : dict
        A dictionary containing shared resources for the cameras, including
        camera objects, frame buffers, and timestamp buffers.
    
    Methods:
    --------
    __init__(shared_resources):
        Initializes the CameraBufferReader with shared resources.
        Parameters:
        -----------
        shared_resources : dict
            A dictionary containing shared resources for the cameras, including
            camera objects, frame buffers, and timestamp buffers.

    read_all():
        Reads frames from all cameras atomically.
        Returns:
        --------
        list of dict
            A list of dictionaries, each containing:
            - 'camera_id': The ID of the camera.
            - 'frame': The frame data as a numpy array.
            - 'timestamp': The timestamp of the frame as a datetime object.
    """

    def __init__(self, shared_resources):
        self._resources = shared_resources
        
    def read_all(self):
        """Read from all cameras atomically"""
        frames = []
        for cam in self._resources['cameras']:
            with cam['lock']:
                frame = np.frombuffer(
                    cam['frame_buffer'].get_obj(),
                    dtype=cam['dtype']
                ).reshape(cam['shape']).copy()
                
                ts_bytes = bytes(cam['timestamp_buffer'].get_obj())
                timestamp = datetime.strptime(
                    ts_bytes.decode('utf-8').strip('\x00'),
                    "%Y%m%d_%H%M%S_%f"
                )
                
            frames.append({
                'camera_id': cam['id'],
                'frame': frame,
                'timestamp': timestamp
            })
        return frames