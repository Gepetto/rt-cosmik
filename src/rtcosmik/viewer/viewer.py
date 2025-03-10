from rtcosmik.config_loader import settings
if settings.viewer == 'ros':
    from .ros_viewer import ros_init, publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
else: # default to gepetto viewer
    from .gv_viewer import gv_init, place_objects
from multiprocessing import Process, Queue, Event
from rtcosmik.human_model.urdf_model import Robot
from typing import List
import numpy as np
class Viewer:
    def __init__(self, model, geom_model, visual_model, keypoint_names, marker_names, freeflyer=False):
        self.model = model
        self.geom_model = geom_model 
        self.visual_model = visual_model
        self.keypoint_names = keypoint_names
        self.marker_names = marker_names
        self.freeflyer = freeflyer
        self.viewer_type = settings.viewer

        # Gepetto viewer specific
        self.viz = None

        # ROS specific publishers
        self.marker_pub = None
        self.keypoints_pub = None
        self.q_pub = None
        self.br = None
        
        if self.viewer_type == 'ros':
            self.keypoints_pub, self.marker_pub, self.q_pub, self.br = ros_init(self.freeflyer)
        else :
            self.viz = gv_init(self.model, self.geom_model, self.visual_model, self.keypoint_names, self.marker_names)
    
    def display_q(self, q):
        if self.viewer_type == 'ros':
            publish_kinematics(q, self.q_pub, self.model.names, self.br)
        else:
            self.viz.display(q)

    def display_keypoints(self, pos_keypoints_dict):
        if self.viewer_type == 'ros':
            publish_keypoints_as_marker_array(list(pos_keypoints_dict.values()), self.keypoints_pub, pos_keypoints_dict.keys())
        else:
            place_objects(self.viz, self.keypoint_names, pos_keypoints_dict)

    def display_markers(self, pos_markers_dict):
        if self.viewer_type == 'ros':
            publish_augmented_markers(list(pos_markers_dict.values()), self.marker_pub, pos_markers_dict.keys())
        else:
            place_objects(self.viz, self.marker_names, pos_markers_dict)

class ViewerProcess(Process):
    def __init__(self,
                 result_queues: List[Queue],
                 stop_event: Event,
                 freeflyer=False):
        super().__init__()
        self.robot_urdf = settings.urdf_path
        self.package_dir = settings.meshes_path
        self.result_queues = result_queues
        self.stop_event = stop_event
        self.keypoint_names = settings.keypoint_names
        self.marker_names = settings.marker_names
        self.freeflyer = freeflyer
        if self.freeflyer:
            self.freeflyer_ori = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        else:
            self.freeflyer_ori = None

        self.robot = Robot(self.robot_urdf, 
                           self.package_dir, 
                           self.freeflyer, 
                           self.freeflyer_ori)
        
        self.model = self.robot.model
        self.geom_model = self.robot.geom_model
        self.visual_model = self.robot.visual_model

        self.viewer = Viewer(self.model, 
                             self.geom_model, 
                             self.visual_model, 
                             self.keypoint_names, 
                             self.marker_names, 
                             self.freeflyer)

    def run(self):
        try: 
            while not self.stop_event.is_set():
                kpts_dict = self.result_queues[0].get()
                mks_dict = self.result_queues[1].get()
                q = self.result_queues[2].get()

                self.viewer.display_keypoints(kpts_dict)
                self.viewer.display_markers(mks_dict)
                self.viewer.display_q(q)
        finally:
            print("Viewer process stopped")