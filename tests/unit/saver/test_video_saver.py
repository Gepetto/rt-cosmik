# From project root (rt-cosmik)
# PYTHONPATH=src:. python -m unittest discover tests/unit -v

import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import unittest
import tempfile
from src.rtcosmik.saver.video_saver import VideoSaver

class TestVideoSaver(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = self.temp_dir.name
        self.camera_id = 0
        self.frame_size = (720, 1280)
        self.fps = 40

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_video_writer_initialization(self):
        saver = VideoSaver(self.camera_id, self.save_dir, fps=self.fps, frame_size=self.frame_size)
        self.assertTrue(saver._video_writer.isOpened())
        saver.close()

if __name__ == "__main__":
    unittest.main()