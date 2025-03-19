from rtcosmik.config_loader import settings
if settings.viewer == 'ros':
    from .ros_viewer import ros_init, publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
else: # default to gepetto viewer
    from .gv_viewer import gv_init, place_objects, place
from multiprocessing import Process, Queue, Event
import pinocchio as pin 
from rtcosmik.human_model.urdf_model import Robot
from rtcosmik.human_model.pin_model import build_dummy_model
from rtcosmik.saver.csv_saver import CSVSaver
from typing import List
import numpy as np
from collections import OrderedDict

class Viewer:
    def __init__(self, model, geom_model, visual_model, keypoint_names, marker_names, freeflyer=False):
        self.model = model
        self.data = self.model.createData()
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
            pin.framesForwardKinematics(self.model, self.data,q)
            for frame in self.model.frames.tolist():
                M = self.data.oMf[self.model.getFrameId(frame.name)]
                place(self.viz, 'world/'+frame.name,  M)
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
        self.joint_angles_names = settings.joint_angles_names
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
                
                # Build the ordered keypoints dict according to the header order.
                ordered_keypoints = OrderedDict()
                for header_key in self.keypoints_header:
                    # Handle frame keys directly.
                    if header_key.startswith("Frame_"):
                        ordered_keypoints[header_key] = kpts_dict_to_save.get(header_key, None)
                    else:
                        # Check if header already contains a coordinate suffix.
                        if header_key.endswith('_x') or header_key.endswith('_y') or header_key.endswith('_z'):
                            base, comp = header_key.rsplit('_', 1)
                            arr = kpts_dict_to_save.get(base)
                            if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                if comp == "x":
                                    ordered_keypoints[header_key] = float(arr[0])
                                elif comp == "y":
                                    ordered_keypoints[header_key] = float(arr[1])
                                elif comp == "z":
                                    ordered_keypoints[header_key] = float(arr[2])
                            else:
                                ordered_keypoints[header_key] = None
                        else:
                            # If the header does not contain a suffix, assume it's a base key and generate three entries.
                            base = header_key
                            arr = kpts_dict_to_save.get(base)
                            if arr is not None and hasattr(arr, '__getitem__') and len(arr) >= 3:
                                ordered_keypoints[base + '_x'] = float(arr[0])
                                ordered_keypoints[base + '_y'] = float(arr[1])
                                ordered_keypoints[base + '_z'] = float(arr[2])
                            else:
                                ordered_keypoints[base + '_x'] = None
                                ordered_keypoints[base + '_y'] = None
                                ordered_keypoints[base + '_z'] = None

                # Build the ordered markers dict similarly.
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

                # (Optional) Debug prints to verify the ordered dictionaries:
                # print("Ordered keypoints:", ordered_keypoints)
                # print("Ordered markers:", ordered_markers)

                self.csv_saver.save_keypoints(ordered_keypoints)
                self.csv_saver.save_markers(ordered_markers)
                self.csv_saver.save_joint_angles(ordered_joint_angles)

        finally:
            print("Viewer process terminated")