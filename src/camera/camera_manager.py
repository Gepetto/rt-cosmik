import numpy as np
import multiprocessing as mp
from camera.camera import start_camera_process
from camera.cam_utils import list_cameras
from datetime import datetime

class CameraManager:
    """
    Manages multiple camera processes, including initialization, starting, stopping, 
    and retrieving frames from the cameras.
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
        start():
            Starts all camera processes.
        get_frames():
            Retrieves the latest frames and timestamps from all cameras.
        stop():
            Stops all camera processes gracefully.
        __enter__():
            Initializes cameras and starts processes when entering a context.
        __exit__(exc_type, exc_val, exc_tb):
            Stops all camera processes when exiting a context.
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
    
    def start(self):
        """Start all camera processes"""
        if not self._cameras:
            raise RuntimeError("No cameras initialized. Call initialize_cameras() first")
            
        for cam in self._cameras:
            process = mp.Process(
                target=start_camera_process,
                args=(
                    cam['id'],
                    self._width,
                    self._height,
                    self._fps,
                    self._fourcc,
                    cam['frame_buffer'],
                    cam['timestamp_buffer'],
                    cam['lock'],
                    self._barrier,
                    self._stop_event
                )
            )
            self._processes.append(process)
            process.start()
    
    def get_frames(self):
        """Retrieve latest frames and timestamps from all cameras"""
        frames = []
        for cam in self._cameras:
            with cam['lock']:
                # Get frame data
                frame = np.frombuffer(cam['frame_buffer'].get_obj(), 
                                     dtype=np.uint8).reshape(self._height, self._width, 3)
                # Get timestamp
                ts_bytes = bytes(cam['timestamp_buffer'].get_obj())
                timestamp = ts_bytes.decode('utf-8').strip('\x00')
                
            frames.append({
                'camera_id': cam['id'],
                'frame': frame.copy(),
                'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
            })
        return frames
    
    def stop(self):
        """Stop all camera processes gracefully"""
        self._stop_event.set()
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
        print("All camera processes terminated")

    def __enter__(self):
        self.initialize_cameras()
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


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