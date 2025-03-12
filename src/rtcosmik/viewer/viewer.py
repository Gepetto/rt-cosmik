from rtcosmik.config_loader import settings
if settings.viewer == 'ros':
    from .ros_viewer import ros_init, publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
else: # default to gepetto viewer
    from .gv_viewer import gv_init, place_objects
from multiprocessing import Process, Queue, Event
from rtcosmik.human_model.urdf_model import Robot
from rtcosmik.human_model.pin_model import build_dummy_model
from rtcosmik.saver.csv_saver import CSVSaver
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
            place_objects(self.viz, pos_keypoints_dict)

    def display_markers(self, pos_markers_dict):
        if self.viewer_type == 'ros':
            publish_augmented_markers(list(pos_markers_dict.values()), self.marker_pub, pos_markers_dict.keys())
        else:
            place_objects(self.viz, pos_markers_dict)

class ViewerProcess(Process):
    def __init__(self,
                 result_queues: List[Queue],
                 stop_event: Event,
                 num_cameras: int,
                 freeflyer=False):
        super().__init__()
        self.robot_urdf = settings.urdf_path
        self.package_dir = settings.meshes_path
        self.result_queues = result_queues
        self.stop_event = stop_event
        self.num_cameras = num_cameras
        self.keypoints_names = settings.keypoints_names
        self.marker_names = settings.marker_names
        self.joint_angles_names = settings.joint_angle_names
        self.freeflyer = freeflyer
        if self.freeflyer:
            self.freeflyer_ori = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        else:
            self.freeflyer_ori = None

        self.SAVE_CSV = settings.SAVE_CSV
        if self.SAVE_CSV:
            self.SAVE_DIR = settings.SAVE_DIR
            self.frame_counters = []
            for i in range(self.num_cameras):
                self.frame_counters.append('Frame_'+str(i))

            self.keypoints_header = self.frame_counters+self.keypoints_names
            self.markers_header = self.frame_counters+self.marker_names
            self.joint_angles_header = self.frame_counters+self.joint_angles_names

    def run(self):
        # self.robot = Robot(self.robot_urdf, 
        #                    self.package_dir, 
        #                    self.freeflyer, 
        #                    self.freeflyer_ori)
        #
        # self.model = self.robot.model
        # self.geom_model = self.robot.geom_model
        # self.visual_model = self.robot.visual_model
        
        self.model, self.geom_model, _ = build_dummy_model(self.package_dir)
        
        self.visual_model = self.geom_model.copy()

        if self.SAVE_CSV:
            self.csv_saver = CSVSaver(
                self.SAVE_DIR,
                self.keypoints_header,
                self.markers_header,
                self.joint_angles_header
            ) 

        self.viewer = Viewer(self.model, 
                             self.geom_model, 
                             self.visual_model, 
                             self.keypoints_names, 
                             self.marker_names, 
                             self.freeflyer)
        
        try: 
            while not self.stop_event.is_set():
                cam_counters, kpts_dict = self.result_queues[0].get()
                _, mks_dict = self.result_queues[1].get()
                _, q = self.result_queues[2].get()

                print("in viewer, counters are :", cam_counters)
                self.viewer.display_keypoints(kpts_dict)
                self.viewer.display_markers(mks_dict)
                self.viewer.display_q(q)

                kpts_dict_to_save = kpts_dict
                mks_dict_to_save = mks_dict
                q_dict_to_save = {}
                for i in range(len(cam_counters)):
                    kpts_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                    mks_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                    q_dict_to_save['Frame_'+str(i)]=cam_counters[i]
                
                for i in range(len(self.joint_angles_names)):
                    q_dict_to_save[self.joint_angles_names[i]]=q[i]
                
                self.csv_saver.save_keypoints(kpts_dict_to_save)
                self.csv_saver.save_markers(mks_dict_to_save)
                self.csv_saver.save_joint_angles(q_dict_to_save)

        finally:
            print("Viewer process terminated")