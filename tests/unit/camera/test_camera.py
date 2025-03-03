# From project root (rt-cosmik)
# PYTHONPATH=src:. python -m unittest discover tests/unit -v

import unittest
import multiprocessing as mp
import numpy as np
import os
import sys
from datetime import datetime
import time
import cv2

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.camera.camera import Camera

class TestCameraClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create shared resources that can be reused across tests
        cls.frame_shape = (240, 320, 3)  # Smaller resolution for faster tests
        cls.shared_buffer = mp.Array('B', int(np.prod(cls.frame_shape)), lock=False)
        cls.timestamp_buffer = mp.Array('c', 26)  # Timestamp buffer
        cls.lock = mp.Lock()

    def create_camera_process(self, cam_id=0):
        return Camera(
            cam_id=cam_id,
            shared_buffer=self.shared_buffer,
            timestamp_buffer=self.timestamp_buffer,
            lock=self.lock,
            frame_shape=self.frame_shape,
            cam_fps=30,
            cam_fourcc="MJPG"
        )

    def test_initialization(self):
        """Test camera process initialization with valid parameters"""
        cam = self.create_camera_process()
        self.assertEqual(cam.frame_shape, self.frame_shape)
        self.assertEqual(cam.cam_fourcc, "MJPG")
        self.assertEqual(cam.cam_fps, 30)

    def test_shared_buffer_updates(self):
        """Verify frame data gets written to shared memory"""
        cam = self.create_camera_process()
        cam.start()
        
        # Give time for frames to start arriving
        time.sleep(1)
        
        # Check buffer has non-zero data
        with self.lock:
            arr = np.frombuffer(self.shared_buffer.get_obj(), dtype=np.uint8)
            self.assertFalse(np.all(arr == 0))
        
        cam.stop()
        cam.join()

    def test_timestamp_format(self):
        """Verify timestamp format and updates"""
        cam = self.create_camera_process()
        cam.start()
        time.sleep(0.5)  # Allow first frame to process
        
        try:
            # Get timestamp from buffer
            with self.lock:
                ts_bytes = bytes(self.timestamp_buffer[:])
            
            ts_str = ts_bytes.decode('utf-8').strip('\x00')
            
            # Validate format
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                self.assertIsInstance(dt, datetime)
            except ValueError:
                self.fail("Invalid timestamp format")
                
        finally:
            cam.stop()
            cam.join()

    def test_process_termination(self):
        """Test clean shutdown sequence"""
        cam = self.create_camera_process()
        cam.start()
        time.sleep(0.5)
        
        # Send stop signal
        cam.stop()
        cam.join(timeout=2)
        
        self.assertFalse(cam.is_alive(), "Process failed to terminate")

    def test_invalid_timestamp_buffer(self):
        """Test buffer size validation"""
        with self.assertRaises(ValueError):
            invalid_ts_buffer = mp.Array('c', 25)
            Camera(
                cam_id=0,
                shared_buffer=self.shared_buffer,
                timestamp_buffer=invalid_ts_buffer,
                lock=self.lock,
                frame_shape=self.frame_shape
            )

    @unittest.skipIf(not cv2.videoio_registry.hasBackend(cv2.CAP_V4L2), "V4L2 backend not available")
    def test_camera_property_settings(self):
        """Verify camera property initialization (requires physical camera)"""
        cam = self.create_camera_process()
        cam.start()
        time.sleep(0.5)  # Allow initialization
        
        # Check properties were set
        temp_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        actual_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
        
        self.assertEqual(actual_width, self.frame_shape[0])
        self.assertEqual(actual_height, self.frame_shape[1])
        
        cam.stop()
        cam.join()

if __name__ == '__main__':
    unittest.main()