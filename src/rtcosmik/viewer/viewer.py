"""Meshcat drawing for a calibrated human model.

Only the drawing. Recording and its keyboard toggle moved to
:class:`rtcosmik.saver.recorder.Recorder`, and the process that used to own all
three is gone: the pipeline process draws through
:class:`rtcosmik.viewer.async_display.AsyncDisplay` using the model it actually
calibrated, instead of a second one rebuilt from settings defaults.

Note this module no longer imports pynput, so it is importable headless.
"""

import logging

import meshcat
import meshcat.geometry as g
from pinocchio.visualize import MeshcatVisualizer
import numpy as np

LOGGER = logging.getLogger(__name__)

class Viewer:
    def __init__(self, model, collision_model, visual_model, marker_names, freeflyer=True):
        self.model = model
        self.collision_model = collision_model 
        self.visual_model = visual_model

        self.marker_names = marker_names
        self.freeflyer = freeflyer
        
        self.vis = meshcat.Visualizer()
        LOGGER.info(f"[INFO] Meshcat visualizer available here: {self.vis.url()}")
        self.vis_markers = self.vis["markers"]

        self.marker_colors = np.zeros((3,len(self.marker_names)))
        self.marker_colors[0, :] = 1.0  # R
        self.marker_colors[1, :] = 0.0  # G
        self.marker_colors[2, :] = 0.0  # B

        # Init meshcat viewer for human
        # Visualizers
        self.viz_human = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz_human.initViewer(self.vis, open=True)
        
        # Don't delete the whole Meshcat tree: keep '/markers' etc.
        try:
            self.vis["ref"].delete()
        except Exception:
            pass
        self.viz_human.loadViewerModel("ref")

        self.viz_human.viewer["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
        self.viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])  # Same color → flat background

    
    def display_q(self, q):
        self.viz_human.display(q)

    def display_markers(self, pos_markers_dict):
        pts = np.stack(list(pos_markers_dict.values()), axis=0).astype(np.float32)
        self.vis_markers.set_object(
                    g.PointCloud(position=pts.T, color=self.marker_colors, size=0.02)
                )
