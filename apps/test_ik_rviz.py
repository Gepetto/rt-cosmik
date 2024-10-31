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
from pinocchio.visualize import GepettoVisualizer, RVizVisualizer
import numpy as np
from utils.model_utils import build_model_challenge
from utils.ik_utils import RT_IK
from utils.viz_utils import place, Rquat
import time
import rospy
from sensor_msgs.msg import JointState

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


# lstm_dict = result_keypoints[100] | result_markers[100]
lstm_dict = {**result_keypoints[100], **result_markers[100]}

t1 =time.time()
human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)
t2 = time.time()

print("Time to build the model: ", t2-t1)

dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

rospy.init_node('human_rt_ik', anonymous=True)
pub = rospy.Publisher('/human_RT_joint_angles', JointState, queue_size=10)

### IK init 
q = pin.neutral(human_model) # init pos
dt = 1/40
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

### IK calculations
ik_class = RT_IK(human_model, lstm_dict, q, keys_to_track_list, dt)
q = ik_class.solve_ik_sample_casadi()
ik_class._q0=q

# Publish joint angles 
joint_state_msg=JointState()
joint_state_msg.header.stamp=rospy.Time.now()
joint_state_msg.name = dof_names
joint_state_msg.position = q.tolist()
pub.publish(joint_state_msg)
print("kinematics published")

q_to_send=q[7:]

print(q)
input()

for ii in range(100,len(result_markers)): 
    lstm_dict = result_markers[ii]
    ik_class._dict_m= lstm_dict
    q = ik_class.solve_ik_sample_quadprog() 
    ik_class._q0 = q 

    q_to_send=q[7:]

    # Publish joint angles 
    joint_state_msg=JointState()
    joint_state_msg.header.stamp=rospy.Time.now()
    joint_state_msg.name = dof_names
    joint_state_msg.position = q_to_send.tolist()
    pub.publish(joint_state_msg)
    print("kinematics published")
    input()
