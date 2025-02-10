import argparse
import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
import cv2
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import time 
# time.sleep(5)
import pinocchio as pin
from collections import deque
from datetime import datetime

# don't forget to source dependancies
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
import tf2_ros
import pandas as pd
from utils.lstm_v2 import augmentTRC, loadModel
from utils.model_utils import build_model_challenge
from utils.calib_utils import get_cameras_params, load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2
from utils.triangulation_utils import triangulate_points_off
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.ros_utils import publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
from utils.read_write_utils import init_csv, save_3dpos_to_csv, save_q_to_csv, read_mmpose_file
from utils.settings import Settings
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

settings = Settings()
liste_fichiers = [
    '/root/workspace/ros_ws/src/rt-cosmik/output/cam_1.csv',
    '/root/workspace/ros_ws/src/rt-cosmik/output/cam_2.csv'

]
donnees_cameras=[]
for fichier in liste_fichiers :
    donnees_cameras.append(read_mmpose_file(fichier))

uvs=[]   
nombre_points = 26
for donnee_camera in donnees_cameras:
    uvs_camera = np.array([[ligne[2*i], ligne[2*i + 1]] for ligne in donnee_camera for i in range(nombre_points)])
    uvs_camera = uvs_camera.reshape(-1, nombre_points, 2)
    uvs.append(uvs_camera)

K1, D1 = load_cam_params(os.path.join(parent_directory,"config/cam_params/c1_params_color_test_test.yaml"))
K2, D2 = load_cam_params(os.path.join(parent_directory,"config/cam_params/c2_params_color_test_test.yaml"))
R,T = load_cam_to_cam_params(os.path.join(parent_directory,"config/cam_params/c1_to_c2_params_color_test_test.yaml"))
mtxs, dists, projections, rotations, translations = get_cameras_params(K1, D1, K2, D2, R, T)

### Loading camera pose 
cam_R1_world, cam_T1_world = load_cam_pose(os.path.join(parent_directory,'config/cam_params/camera1_pose_test_test.yaml'))
# Inverse the pose to get cam in world frame 
world_R1_cam = cam_R1_world.T
world_T1_cam = -cam_R1_world.T@cam_T1_world
world_T1_cam = world_T1_cam.reshape((3,))

p3d_frame = triangulate_points_off(uvs, mtxs, dists, projections)
keypoints_in_cam = p3d_frame

# Apply the rotation matrix to align the points
keypoints_in_world_list = []  # Store all frames

for frame in keypoints_in_cam:  # Iterate over frames
    keypoints_in_world_frame = []  # Store transformed points for this frame

    for p in frame:  # Iterate over the 26 points in this frame
        transformed_p = np.dot(world_R1_cam, p) + world_T1_cam  # Apply transformation
        keypoints_in_world_frame.append(transformed_p)  # Store transformed point

    keypoints_in_world_frame = np.array(keypoints_in_world_frame).flatten()  # Flatten (26,3) → (78,)
    keypoints_in_world_list.append(keypoints_in_world_frame)  # Store the frame
print(len(keypoints_in_world_list))

# Convert to DataFrame
df = pd.DataFrame(keypoints_in_world_list)

output_csv_path = "/root/workspace/ros_ws/src/rt-cosmik/output/keypoints_3d.csv"

# Save to CSV without header/index
df.to_csv(output_csv_path, index=False, header=False)

print(f"Saved {len(keypoints_in_world_list)} frames to {output_csv_path}")