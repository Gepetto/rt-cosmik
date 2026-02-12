import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.visualize import MeshcatVisualizer
    
from multiprocessing import Process, Queue, Event
import pinocchio as pin 
import example_robot_data as robex
from src.rtcosmik.saver.csv_saver import CSVSaver
from src.rtcosmik.human_model.model_utils import scale_human_model
from typing import List
import numpy as np
from collections import OrderedDict

from pynput import keyboard
import logging

LOGGER = logging.getLogger(__name__)

class Viewer:
    def __init__(self, settings, model, collision_model, visual_model, marker_names, freeflyer=True):
        self.model = model
        self.collision_model = collision_model 
        self.visual_model = visual_model

        self.marker_names = marker_names
        self.freeflyer = freeflyer
        self.viewer_type = settings.viewer
        
        self.vis = meshcat.Visualizer()
        LOGGER.info(f"[INFO] Meshcat visualizer available here: {self.vis.url()}")
        self.vis_markers = vis["markers"]

        self.marker_colors = np.zeros_like(self.marker_names)
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

        # ROS specific publishers
        self.marker_pub = None
        self.q_pub = None
        self.br = None
        
        if self.viewer_type == 'ros':
            from .ros_viewer import ros_init, publish_augmented_markers, publish_kinematics
            self.keypoints_pub, self.marker_pub, self.q_pub, self.br = ros_init(self.freeflyer)
    
    def display_q(self, q):
        if self.viewer_type == 'ros':
            publish_kinematics(q, self.q_pub, self.model.names, self.br)
        else:
            self.viz_human.display(q)

    def display_markers(self, pos_markers_dict):
        if self.viewer_type == 'ros':
            publish_augmented_markers(list(pos_markers_dict.values()), self.marker_pub, pos_markers_dict.keys())
        else:
            self.vis_markers.set_object(
                    g.PointCloud(position=pos_markers_dict.values().T, color=self.marker_colors, size=0.02)
                )

class ViewerProcess(Process):
    def __init__(self,
                 settings,
                 result_queues: List[Queue],
                 stop_event: Event,
                 num_cameras: int,
                 freeflyer=True,
                 saving_flag=None,
                 logger=None,
                 ):
        super().__init__()
        self.settings=settings

        self.saving_flag = saving_flag

        self.result_queues = result_queues
        self.stop_event = stop_event
        self.num_cameras = num_cameras
        self.marker_names = self.settings.marker_names
        self.joint_angles_names = self.settings.joint_angles_names

        self.freeflyer = freeflyer
        self.first_sample=True
        self.logger = logger or LOGGER

        self.SAVE_CSV = self.settings.SAVE_CSV
        if self.SAVE_CSV:
            self.SAVE_DIR = self.settings.SAVE_DIR
            self.frame_counters = []
            for i in range(self.num_cameras):
                self.frame_counters.append('Frame_'+str(i))

            self.markers_header = self.frame_counters+self.marker_names
            self.joint_angles_header = self.frame_counters+self.joint_angles_names
            self.logger.info("[INFO] Saving data as csv enabled ...")

    def run(self):

        #load model from AT
        self.human = robex.human.HumanLoader(height=self.settings.human_height, weight=self.settings.human_weight, gender=self.settings.human_gender).robot
        self.model= self.human.model
        self.collision_model = self.human.collision_model
        self.visual_model  = self.human.visual_model
        self.logger.info("[INFO] Human model successfully loaded in viewer process...")

        self.saving_enabled = False

        if self.SAVE_CSV:
            self.csv_saver = CSVSaver(
                self.SAVE_DIR,
                self.keypoints_header,
                self.markers_header,
                self.joint_angles_header
            ) 

        self.viewer = Viewer(
                             self.settings,
                             self.model, 
                             self.geom_model, 
                             self.visual_model, 
                             self.marker_names, 
                             self.freeflyer
                             )
        
        def on_press(key):
            try:
                if key.char == 's':
                    self.logger.info("[INFO] Start saving data in viewer process ...")
                    self.saving_enabled = True
                    if self.saving_flag is not None:
                        self.saving_flag.value = True  
                elif key.char == 'q':
                    self.logger.info("[INFO] Stop saving data in viewer process ...")
                    self.saving_enabled = False
                    if self.saving_flag is not None:
                        self.saving_flag.value = False  
            except AttributeError:
                pass

        # Start keyboard listener in background
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        try: 
            while not self.stop_event.is_set():
                cam_counters, mks_dict = self.result_queues[0].get()
                _, q = self.result_queues[1].get()

                #scale the model to data
                if self.first_sample:
                    self.model = scale_human_model(self.model, mks_dict, gender=self.settings.human_gender, subject_height=self.settings.human_height)
                    self.first_sample=False
                
                self.viewer.display_markers(mks_dict)
                self.viewer.display_q(q)

                if self.SAVE_CSV and self.saving_enabled:
                    mks_dict_to_save = mks_dict
                    q_dict_to_save = {}
                    for i in range(len(cam_counters)):
                        mks_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                        q_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                    
                    for i in range(len(self.joint_angles_names)):
                        q_dict_to_save[self.joint_angles_names[i]]=q[i]
                    
                    ordered_markers = OrderedDict()
                    for header_key in self.markers_header:
                        if header_key.startswith("Frame_"):
                            ordered_markers[header_key] = mks_dict_to_save.get(header_key, None)
                        else:
                            if header_key.endswith('_x') or header_key.endswith('_y') or header_key.endswith('_z'):
                                base, comp = header_key.rsplit('_', 1)
                                arr = mks_dict_to_save.get(base)
                                if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                    if comp == "x":
                                        ordered_markers[header_key] = float(arr[0])
                                    elif comp == "y":
                                        ordered_markers[header_key] = float(arr[1])
                                    elif comp == "z":
                                        ordered_markers[header_key] = float(arr[2])
                                else:
                                    ordered_markers[header_key] = None
                            else:
                                base = header_key
                                arr = mks_dict_to_save.get(base)
                                if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                    ordered_markers[base + '_x'] = float(arr[0])
                                    ordered_markers[base + '_y'] = float(arr[1])
                                    ordered_markers[base + '_z'] = float(arr[2])
                                else:
                                    ordered_markers[base + '_x'] = None
                                    ordered_markers[base + '_y'] = None
                                    ordered_markers[base + '_z'] = None

                    # Joint angles are assumed to be scalars.
                    ordered_joint_angles = OrderedDict(
                        (key, q_dict_to_save.get(key, None)) for key in self.joint_angles_header
                    )

                    self.csv_saver.save_markers(ordered_markers)
                    self.csv_saver.save_joint_angles(ordered_joint_angles)

        finally:
            listener.stop()
            self.logger.info("[INFO] Viewer process terminated")
