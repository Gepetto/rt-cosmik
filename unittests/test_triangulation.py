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
from datetime import datetime

# don't forget to source dependancies
import pandas as pd
from utils.calib_utils import get_cameras_params, load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2
from utils.triangulation_utils import  triangulate_points
from utils.read_write_utils import init_csv, save_3dpos_to_csv, save_q_to_csv, read_mmpose_file
from utils.settings import Settings
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

liste_fichiers = [
    '/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/cam_1.csv',
    '/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/cam_2.csv'

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

keypoints_in_world_list = []  # Store all frames
num_frames = len(uvs[0])  # Nombre de frames, basé sur la première caméra

for frame_idx in range(num_frames):
    points_2d_per_frame = [uv[frame_idx] for uv in uvs]
    
    p3d_frame = triangulate_points(points_2d_per_frame, mtxs, dists, projections)
    keypoints_in_cam = p3d_frame

    keypoints_in_world_frame = []  # Store transformed points for this frame

# Apply the rotation matrix to align the points
    for point in keypoints_in_cam:  # Iterate over frames
        keypoints_in_world = np.dot(world_R1_cam, point) + world_T1_cam  # Apply transformation
        keypoints_in_world_frame.append(keypoints_in_world)

    keypoints_in_world_list.append(np.array(keypoints_in_world_frame).flatten().tolist())


# Convert to DataFrame
df = pd.DataFrame(keypoints_in_world_list)

output_csv_path = "/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/keypoints_3d.csv"

# Save to CSV without header/index
df.to_csv(output_csv_path, index=False, header=False)

print(f"Saved {len(keypoints_in_world_list)} frames to {output_csv_path}")