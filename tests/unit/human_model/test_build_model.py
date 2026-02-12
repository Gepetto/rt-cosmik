import sys
import os

cosmik_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, cosmik_path) # Repo root
sys.path.insert(0, os.path.join(cosmik_path, "src")) # src dir

import unittest
import pinocchio as pin
import numpy as np
import pandas as pd
import time

# Adjust the import path according to your project structure.
from rtcosmik.human_model.pin_model import build_model_no_visuals
from rtcosmik.human_model.model_utils import construct_segments_frames
from rtcosmik.viewer.gv_viewer import place

class TestBuildModelNoVisuals(unittest.TestCase):
    def setUp(self):
        """
        Load markers.csv and keypoints.csv.
        
        For markers.csv:
         - Skip the first two columns ('Frame_0' and 'Frame_1').
         - For every remaining column ending with '_x', group with the corresponding '_y' and '_z' columns.
        
        For keypoints.csv:
         - Only extract keypoints: 'Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye'.
         - Each keypoint is expected in columns like Nose_x, Nose_y, Nose_z, etc.
        """
        # Load markers.csv
        markers_df = pd.read_csv(os.path.join(cosmik_path,"tests","full","data","markers.csv"))
        markers_row = markers_df.iloc[0]  # Use the first row for testing
        
        # Build markers dictionary: iterate over columns from index 2 onward.
        self.mocap_mks_positions = {}
        for col in markers_df.columns[2:]:
            if col.endswith('_x'):
                marker_name = col[:-2]  # Remove '_x' suffix to get marker name
                # Look for corresponding y and z columns
                col_y = marker_name + '_y'
                col_z = marker_name + '_z'
                if col_y in markers_df.columns and col_z in markers_df.columns:
                    x = markers_row[col]
                    y = markers_row[col_y]
                    z = markers_row[col_z]
                    # Only add if values are not NaN
                    if pd.notna(x) and pd.notna(y) and pd.notna(z):
                        self.mocap_mks_positions[marker_name] = np.array([x, y, z])
        
        # Load keypoints.csv
        keypoints_df = pd.read_csv(os.path.join(cosmik_path,"tests","full","data","keypoints.csv"))
        keypoints_row = keypoints_df.iloc[0]  # Use the first row for testing
        extra_keypoints = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
        for kp in extra_keypoints:
            col_x = kp + '_x'
            col_y = kp + '_y'
            col_z = kp + '_z'
            if col_x in keypoints_df.columns and col_y in keypoints_df.columns and col_z in keypoints_df.columns:
                x = keypoints_row[col_x]
                y = keypoints_row[col_y]
                z = keypoints_row[col_z]
                if pd.notna(x) and pd.notna(y) and pd.notna(z):
                    self.mocap_mks_positions[kp] = np.array([x, y, z])
            else:
                self.fail(f"Missing columns for keypoint: {kp}")

    def test_model_generation(self):
        """Test that the Pinocchio model is generated correctly."""
        model = build_model_no_visuals(self.mocap_mks_positions)
        
        # Check that the returned object is a Pinocchio Model
        self.assertIsInstance(model, pin.Model)
        # Check that the model contains at least one joint
        self.assertGreater(model.njoints, 0)
        
        # Check that some expected frames (e.g., 'pelvis', 'torso', 'head') are present in the model frames.
        frame_names = [frame.name for frame in model.frames]
        expected_frames = construct_segments_frames(self.mocap_mks_positions).keys()
        for expected in expected_frames:
            self.assertTrue(
                any(expected in name for name in frame_names),
                f"Expected frame '{expected}' not found in model frames: {frame_names}"
            )

    def test_viewer(self):
        """
        Launch Gepetto viewer for manual verification of the model.
        
        This test adds a small red sphere at each frame location.
        Note: It will pause for 10 seconds. If the viewer or display is unavailable,
        this test will be skipped.
        """
        model = build_model_no_visuals(self.mocap_mks_positions)
        try:
            from pinocchio.visualize import GepettoVisualizer
        except ImportError:
            self.skipTest("Gepetto Viewer is not available in this environment.")
        
        viz = GepettoVisualizer(model, None, None)
        try:
            viz.initViewer()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install gepetto-viewer"
            )
            print(err)
            sys.exit(0)

        try:
            viz.loadViewerModel("pinocchio")
        except AttributeError as err:
            print(
                "Error while loading the viewer model. It seems you should start gepetto-viewer"
            )
            print(err)
            sys.exit(0)
            
        # Add a XYZ axis at each frame for visual inspection.
        # Init objects to show 
        # Frame axis for frame in the pinocchio model 
        for frame in model.frames.tolist():
            viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

        q = pin.neutral(model)
        data = model.createData()
        pin.framesForwardKinematics(model, data, q)

        for frame in model.frames.tolist():
            M = data.oMf[model.getFrameId(frame.name)]
            place(viz, 'world/'+frame.name,  M)
        
        print("Gepetto viewer launched. Please inspect the viewer window for 10 seconds.")
        time.sleep(10)
        # After manual inspection, the test ends.

if __name__ == "__main__":
    unittest.main()
