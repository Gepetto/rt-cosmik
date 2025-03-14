# From project root (rt-cosmik)
# PYTHONPATH=src:. python -m unittest discover tests/unit -v

import unittest
from unittest.mock import Mock, patch
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
        # Use standard VGA resolution
        cls.frame_shape = (480, 640, 3)  # (height, width, channels)
        buffer_size = int(np.prod(cls.frame_shape))
        cls.shared_buffer = mp.Array('B', buffer_size, lock=False)
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
        
        try:
            # Wait longer with progressive checks
            start_time = time.time()
            updated = False
            
            while time.time() - start_time < 10:  # 10-second timeout
                with self.lock:
                    arr = np.frombuffer(self.shared_buffer, dtype=np.uint8)
                    if np.any(arr != 0):
                        updated = True
                        break
                time.sleep(0.2)
            
            self.assertTrue(updated, "Shared buffer never received data")
        finally:
            cam.stop()
            cam.join()

    def test_timestamp_format(self):
        cam = self.create_camera_process()
        cam.start()
        
        try:
            # Wait for valid timestamp
            start_time = time.time()
            ts_str = ""
            
            while time.time() - start_time < 10:  # 10-second timeout
                with self.lock:
                    ts_bytes = bytes(self.timestamp_buffer[:])
                    ts_str = ts_bytes.decode('utf-8').split('\x00')[0]
                    if ts_str:
                        break
                time.sleep(0.2)
            
            self.assertGreater(len(ts_str), 23, f"Invalid timestamp: '{ts_str}'")
            datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
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

    @unittest.skipIf(not cv2.VideoCapture(0).isOpened(), "No camera available")
    def test_camera_property_settings(self):
        """Verify camera property initialization"""
        # Skip if no camera available
        if not cv2.VideoCapture(0).isOpened():
            self.skipTest("No camera detected at index 0")
        
        # Use camera's native resolution
        test_shape = (480, 640, 3)  # (height, width, channels)
        
        # Create camera with test shape
        cam = Camera(
            cam_id=0,
            shared_buffer=self.shared_buffer,
            timestamp_buffer=self.timestamp_buffer,
            lock=self.lock,
            frame_shape=test_shape,
            cam_fps=30
        )
        
        try:
            cam.start()
            time.sleep(3)  # Longer warmup for hardware initialization
            
            # Verify actual properties
            cap = cv2.VideoCapture(0)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Check if camera accepted our settings
            if actual_width == 0 or actual_height == 0:
                self.skipTest("Camera properties not readable")
                
            # Allow 10% tolerance for resolution mismatch
            self.assertAlmostEqual(actual_width, test_shape[1], delta=test_shape[1]*0.1)
            self.assertAlmostEqual(actual_height, test_shape[0], delta=test_shape[0]*0.1)
            
        finally:
            cam.stop()
            cam.join()

@unittest.skipIf(cv2.VideoCapture(0).isOpened(), "Skipping mock tests when real camera is available")
class TestMockCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame_shape = (480, 640, 3)  # Match your expected shape
        cls.shared_buffer = mp.Array('B', int(np.prod(cls.frame_shape)), lock=False)
        cls.timestamp_buffer = mp.Array('c', 26)
        cls.lock = mp.Lock()

    @patch('cv2.VideoCapture')
    def test_mocked_camera_operation(self, mock_videocapture):
        # Configure mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.random.randint(0, 255, self.frame_shape, dtype=np.uint8))
        mock_videocapture.return_value = mock_cap

        # Create and run camera
        cam = Camera(
            cam_id=0,
            shared_buffer=self.shared_buffer,
            timestamp_buffer=self.timestamp_buffer,
            lock=self.lock,
            frame_shape=self.frame_shape
        )
        
        cam.start()
        time.sleep(0.5)  # Allow frame capture
        cam.stop()
        cam.join()

        # Verify buffer updates
        with self.lock:
            arr = np.frombuffer(self.shared_buffer, dtype=np.uint8)
            self.assertFalse(np.all(arr == 0), "Buffer should contain image data")

if __name__ == '__main__':
    unittest.main()