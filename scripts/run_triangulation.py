#Triangulate from 2 csv files of 2dkeypoints
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import numpy as np
import pandas as pd
from src.rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation,load_four_camera_parameters
from src.rtcosmik.triangulation.triangulation import triangulate_offline,triangulate_points_adaptive
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv,load_transformation,transform_keypoints_list_cam0_to_mocap,read_mmpose_scores
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter
#check paths in load_camera_parameters and load_world_transformation
no_trial = "Maxime"
task = "static"

transformation_file = f"/root/workspace/ros_ws/src/rt-cosmik/config/cam_params/{no_trial}/calib_mocap_2_cam0/soder.txt"
R_trans, d_trans, s_trans, rms_error = load_transformation(transformation_file)

num_keypoints=26 
markers = [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ]
header = []
for marker in markers:
    header.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

def main():
    base_path = "/root/workspace/ros_ws/src/rt-cosmik"
    config_path = os.path.join(base_path, "config/cam_params/Maxime")
    output_csv_path = os.path.join(base_path, f"output/{no_trial}/{task}/3d_keypoints_filtred_mocap.csv")
    file_paths = [
        os.path.join(base_path, f"output/{no_trial}/output_2d/{task}/{task}_camera_0.csv"),
        os.path.join(base_path, f"output/{no_trial}/output_2d/{task}/{task}_camera_2.csv")
    ]
    
    camera_data = [read_mmpose_file(file) for file in file_paths]
    uvs = [
        np.array([[line[2 * i], line[2 * i + 1]] for line in data for i in range(num_keypoints)])
        .reshape(-1, num_keypoints, 2)
        for data in camera_data
    ]

    mtxs, dists, projections, rotations, translations = load_camera_parameters(config_path)
    # world_R1_cam, world_T1_cam = load_world_transformation(config_path)
    
    # keypoints_in_cam0_list = triangulate_offline(uvs, mtxs, dists, projections, world_R1_cam, world_T1_cam)
    scores = read_mmpose_scores(file_paths)
    threshold = 0.0
    keypoints_in_cam0_list = triangulate_points_adaptive(uvs, mtxs, dists, projections, scores, threshold)

    keypoints_in_mocap = transform_keypoints_list_cam0_to_mocap(
        keypoints_in_cam0_list,
        R_trans,
        d_trans
    )
    filtered_data = butterworth_filter(
    data=keypoints_in_mocap,
    cutoff_frequency=10.0,  
    order=5,
    sampling_frequency=40
    )
    save_to_csv(filtered_data, output_csv_path, header=header)

if __name__ == "__main__":
    main()