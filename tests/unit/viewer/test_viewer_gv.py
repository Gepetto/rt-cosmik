# From project root (rt-cosmik)
# PYTHONPATH=src:. python -m unittest discover tests/unit -v

import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest

# --- Dummy Settings and Models for Testing --- #

class DummySettings:
    viewer = 'gv'  # Force Gepetto viewer branch

class DummyModel:
    def __init__(self):
        self.names = ['joint1', 'joint2', 'joint3']

dummy_model = DummyModel()
dummy_geom_model = object()  # Dummy placeholder for Pinocchio geom model
dummy_visual_model = object()  # Dummy placeholder for Pinocchio visual model
dummy_keypoint_names = ['kp1', 'kp2']
dummy_marker_names = ['mk1', 'mk2']

# --- Fake Implementations for Gepetto Viewer Functions --- #

class FakeGV:
    def __init__(self):
        self.display_called = False
        self.last_q = None
        self.place_objects_called = False
        self.last_names = None
        self.last_pos_dict = None

    def display(self, q):
        self.display_called = True
        self.last_q = q

def fake_gv_init(model, geom_model, visual_model, keypoint_names, marker_names):
    fake_gv_init.called = True
    fake_gv_init.args = (model, geom_model, visual_model, keypoint_names, marker_names)
    return FakeGV()

fake_gv_init.called = False
fake_gv_init.args = None

def fake_place_objects(viz, names, pos_dict):
    viz.place_objects_called = True
    viz.last_names = names
    viz.last_pos_dict = pos_dict

# --- Patch the Viewer Module --- #
# Import the viewer module now that the sys.path is adjusted.
from src.rtcosmik.viewer.viewer import Viewer

# Override settings and viewer-specific functions in the viewer module.
import src.rtcosmik.viewer.viewer as viewer_module
viewer_module.settings = DummySettings()
viewer_module.gv_init = fake_gv_init
viewer_module.place_objects = fake_place_objects

# --- Unit Tests for the Viewer Class in Gepetto Mode --- #

class TestViewerGV(unittest.TestCase):

    def setUp(self):
        # Reset fake_gv_init flags before each test.
        fake_gv_init.called = False
        fake_gv_init.args = None

    def test_viewer_initialization(self):
        """Verify that in gv mode the Viewer initializes correctly by calling gv_init."""
        v = Viewer(dummy_model, dummy_geom_model, dummy_visual_model,
                   dummy_keypoint_names, dummy_marker_names, freeflyer=False)
        self.assertTrue(fake_gv_init.called)
        m, gm, vm, kp_names, mk_names = fake_gv_init.args
        self.assertEqual(m, dummy_model)
        self.assertEqual(gm, dummy_geom_model)
        self.assertEqual(vm, dummy_visual_model)
        self.assertEqual(kp_names, dummy_keypoint_names)
        self.assertEqual(mk_names, dummy_marker_names)
        self.assertTrue(hasattr(v._viz, 'display'))
        self.assertIsNone(v.keypoints_pub)
        self.assertIsNone(v.marker_pub)
        self.assertIsNone(v.q_pub)
        self.assertIsNone(v.br)

    def test_display_q(self):
        """Test that display_q delegates to the FakeGV.display method."""
        v = Viewer(dummy_model, dummy_geom_model, dummy_visual_model,
                   dummy_keypoint_names, dummy_marker_names, freeflyer=False)
        test_q = [0.1, 0.2, 0.3]
        v.display_q(test_q)
        self.assertTrue(v.viz.display_called)
        self.assertEqual(v.viz.last_q, test_q)

    def test_display_keypoints(self):
        """Test that display_keypoints calls place_objects with the correct arguments."""
        v = Viewer(dummy_model, dummy_geom_model, dummy_visual_model,
                   dummy_keypoint_names, dummy_marker_names, freeflyer=False)
        dummy_keypoints = {'kp1': [1, 2, 3], 'kp2': [4, 5, 6]}
        v.display_keypoints(dummy_keypoints)
        self.assertTrue(v.viz.place_objects_called)
        self.assertEqual(v.viz.last_names, dummy_keypoint_names)
        self.assertEqual(v.viz.last_pos_dict, dummy_keypoints)

    def test_display_markers(self):
        """Test that display_markers calls place_objects with the correct arguments."""
        v = Viewer(dummy_model, dummy_geom_model, dummy_visual_model,
                   dummy_keypoint_names, dummy_marker_names, freeflyer=False)
        dummy_markers = {'mk1': [7, 8, 9], 'mk2': [10, 11, 12]}
        v.display_markers(dummy_markers)
        self.assertTrue(v.viz.place_objects_called)
        self.assertEqual(v.viz.last_names, dummy_marker_names)
        self.assertEqual(v.viz.last_pos_dict, dummy_markers)

if __name__ == '__main__':
    unittest.main()
