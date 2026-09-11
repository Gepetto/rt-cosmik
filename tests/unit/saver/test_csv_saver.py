# From project root (rt-cosmik)
# PYTHONPATH=src:. python -m unittest discover tests/unit -v

import os
import sys 
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import unittest
import os
import tempfile
from collections import OrderedDict
from rtcosmik.saver.csv_saver import CSVSaver

class TestCSVSaver(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = self.temp_dir.name
        self.keypoints_header = ["Frame", "Nose", "RShoulder", "RFoot"]
        self.markers_header = ["Frame", "RPSI_study", "LASI_study", "RASI_study"]
        self.joint_angles_header = ["Frame", "FF_x", "FF_y", "FF_z"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_csv_headers(self):
        with CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header) as saver:
            # Ensure files are created
            self.assertTrue(os.path.exists(saver.keypoints_path))
            self.assertTrue(os.path.exists(saver.markers_path))
            self.assertTrue(os.path.exists(saver.joint_angles_path))

            # Expected headers based on the CSVSaver logic
            expected_keypoints_header = "Frame,Nose_x,Nose_y,Nose_z,RShoulder_x,RShoulder_y,RShoulder_z,RFoot_x,RFoot_y,RFoot_z"
            expected_markers_header = "Frame,RPSI_study_x,RPSI_study_y,RPSI_study_z,LASI_study_x,LASI_study_y,LASI_study_z,RASI_study_x,RASI_study_y,RASI_study_z"
            expected_joint_angles_header = "Frame,FF_x,FF_y,FF_z"

            with open(saver.keypoints_path, 'r') as f:
                header = f.readline().strip()
                self.assertEqual(header, expected_keypoints_header)

            with open(saver.markers_path, 'r') as f:
                header = f.readline().strip()
                self.assertEqual(header, expected_markers_header)

            with open(saver.joint_angles_path, 'r') as f:
                header = f.readline().strip()
                self.assertEqual(header, expected_joint_angles_header)

    def test_ordered_dict_requirement(self):
        saver = CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header)
        with self.assertRaises(ValueError):
            saver.save_keypoints({"x": 1, "y": 2})  # Regular dict, not OrderedDict
        saver.close()

    def test_save_keypoints(self):
        saver = CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header)
        keypoints_data = OrderedDict([("Frame", 1), ("Nose_x", 0.1), ("Nose_y", 0.2), ("Nose_z", 0.3),
                                      ("RShoulder_x", 0.4), ("RShoulder_y", 0.5), ("RShoulder_z", 0.6),
                                      ("RFoot_x", 0.7), ("RFoot_y", 0.8), ("RFoot_z", 0.9)])
        saver.save_keypoints(keypoints_data)
        saver.close()

        with open(saver.keypoints_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 data row
            self.assertEqual(lines[1].strip(), "1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")

    def test_save_markers(self):
        saver = CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header)
        markers_data = OrderedDict([("Frame", 1), ("RPSI_study_x", 0.1), ("RPSI_study_y", 0.2), ("RPSI_study_z", 0.3),
                                    ("LASI_study_x", 0.4), ("LASI_study_y", 0.5), ("LASI_study_z", 0.6),
                                    ("RASI_study_x", 0.7), ("RASI_study_y", 0.8), ("RASI_study_z", 0.9)])
        saver.save_markers(markers_data)
        saver.close()

        with open(saver.markers_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 data row
            self.assertEqual(lines[1].strip(), "1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")

    def test_save_joint_angles(self):
        saver = CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header)
        joint_angles_data = OrderedDict([("Frame", 1), ("FF_x", 0.1), ("FF_y", 0.2), ("FF_z", 0.3)])
        saver.save_joint_angles(joint_angles_data)
        saver.close()

        with open(saver.joint_angles_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 data row
            self.assertEqual(lines[1].strip(), "1,0.1,0.2,0.3")

    def test_file_closing(self):
        saver = CSVSaver(self.save_dir, self.keypoints_header, self.markers_header, self.joint_angles_header)
        saver.close()

        self.assertTrue(saver.keypoints_file.closed)
        self.assertTrue(saver.markers_file.closed)
        self.assertTrue(saver.joint_angles_file.closed)

if __name__ == "__main__":
    unittest.main()
