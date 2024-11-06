import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')

import pandas as pd 
import pinocchio as pin 
import numpy as np
from utils.model_utils import build_model_challenge
from utils.ik_utils import RT_SWIKA
from utils.viz_utils import place, Rquat
from utils.ros_utils import publish_kinematics, publish_augmented_markers, publish_keypoints_as_marker_array
import time
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import tf2_ros
from collections import deque

deque_x = deque(maxlen=30)
deque_q = deque(maxlen=30)
deque_lstm_dict = deque(maxlen=30)

rospy.init_node('human_rt_ik', anonymous=True)
pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)
augmented_markers_pub = rospy.Publisher('/markers_pose', MarkerArray, queue_size=10)
keypoints_pub = rospy.Publisher('/pose_keypoints', MarkerArray, queue_size=10)
    
data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/augmented_markers_positions.csv'))
data_keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))

result_markers = []
for frame, group in data_markers.groupby("Frame"):
    frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    result_markers.append(frame_dict)

result_keypoints = []
for frame, group in data_keypoints.groupby("Frame"):
    frame_dict = {row["Keypoint"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    if "Hip" in frame_dict:
        frame_dict["midHip"] = frame_dict.pop("Hip")
    result_keypoints.append(frame_dict)

publish_keypoints_as_marker_array(list(result_keypoints[0].values()), keypoints_pub, list(result_keypoints[0].keys()))
publish_augmented_markers(list(result_markers[0].values()), augmented_markers_pub, list(result_markers[0].keys()))


lstm_dict = {**result_keypoints[0], **result_markers[0]}

t1 =time.time()
human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)
t2 = time.time()

print("Time to build the model: ", t2-t1)

dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

### IK init 
dt = 1/40
T=30
keys_to_track_list = ['C7_study',
                        'r.ASIS_study', 'L.ASIS_study', 
                        'r.PSIS_study', 'L.PSIS_study', 
                        
                        'r_shoulder_study',
                        'r_lelbow_study', 'r_melbow_study',
                        'r_lwrist_study', 'r_mwrist_study',
                        'r_ankle_study', 'r_mankle_study',
                        'r_toe_study','r_5meta_study', 'r_calc_study',
                        'r_knee_study', 'r_mknee_study',
                        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
                        'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
                        
                        'L_shoulder_study', 
                        'L_lelbow_study', 'L_melbow_study',
                        'L_lwrist_study','L_mwrist_study',
                        'L_ankle_study', 'L_mankle_study', 
                        'L_toe_study','L_5meta_study', 'L_calc_study',
                        'L_knee_study', 'L_mknee_study',
                        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
                        'L_sh1_study', 'L_sh2_study', 'L_sh3_study']

x0 = np.zeros(human_model.nv + human_model.nv)
q0 = np.zeros(human_model.nq)
for ii in range(T+1):
    deque_x.append(x0)
    deque_q.append(q0)
    deque_lstm_dict.append(lstm_dict)

### IK calculations
ik_class = RT_SWIKA(human_model, deque_lstm_dict, deque_x, deque_q, keys_to_track_list, T, dt)
X = ik_class.solve_swika_casadi()
