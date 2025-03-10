import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import numpy as np
import pandas as pd
from src.rtcosmik.camera.cam_utils import load_camera_parameters,load_world_transformation
from src.rtcosmik.triangulation.triangulation import triangulate_offline
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv

#check paths in load_camera_parameters and load_world_transformation
num_keypoints=26 

def main():
    base_path = "/root/workspace/ros_ws/src/rt-cosmik"
    config_path = os.path.join(base_path, "config/cam_params")
    output_csv_path = os.path.join(base_path, "output/keypoints_3d_test.csv")
    file_paths = [
        os.path.join(base_path, "output/frontal_plan/cam_1.csv"),
        os.path.join(base_path, "output/frontal_plan/cam_2.csv")
    ]
    
    camera_data = [read_mmpose_file(file) for file in file_paths]
    uvs = [
        np.array([[line[2 * i], line[2 * i + 1]] for line in data for i in range(num_keypoints)])
        .reshape(-1, num_keypoints, 2)
        for data in camera_data
    ]

    mtxs, dists, projections, rotations, translations = load_camera_parameters(config_path)
    world_R1_cam, world_T1_cam = load_world_transformation(config_path)
    
    keypoints_in_world = triangulate_offline(uvs, mtxs, dists, projections, world_R1_cam, world_T1_cam)
    # save_to_csv(keypoints_in_world, output_csv_path)

if __name__ == "__main__":
    main()