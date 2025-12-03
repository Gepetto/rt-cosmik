import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from src.rtcosmik.utils.read_write_utils import read_mks_data
import pandas as pd


SUBJECTS = [
     "Emmanuelle"
]

tasks = ['bolting', 'sanding','overhead', 'robot_sanding', 'robot_welding', 'lifting']

# subject_mocap = "zoe"
# s = "Zoe"
# task = "robot_welding" 
start_sample = 0 

for subject_mocap in SUBJECTS:
    for task in tasks: 
        base_path = f"/root/workspace/ros_ws/src/rt-cosmik/output"
        path = f"{base_path}/mocap_jcp/{subject_mocap}"
        path_to_csv_jcp = f"{base_path}/mocap_jcp/{subject_mocap}/{task}/joint_center_positions_with_offsets.csv"
        df_jcp_mocap = pd.read_csv(path_to_csv_jcp) #jcp mocap
        result_jcp_mocap, start_sample_jcp = read_mks_data(df_jcp_mocap, start_sample=start_sample,converter = 1.0) #check the function of read 

        path_to_csv_mks = f"{base_path}/mocap/mocap_{subject_mocap}/{task}/mocap_downsampled_to_40hz.csv"
        df_mks_mocap = pd.read_csv(path_to_csv_mks) #jcp mocap
        result_mks_mocap, start_sample_mks = read_mks_data(df_mks_mocap, start_sample=start_sample,converter = 1.0) #check the function of read 

        markers_to_add =["FHD_x", "FHD_y","FHD_z","LHD_x", "LHD_y","LHD_z", "RHD_x", "RHD_y","RHD_z"]
        for m in markers_to_add:
            if m in df_mks_mocap.columns:
                df_jcp_mocap[m] = df_mks_mocap[m]/1000
            else:
                print(f"Warning: {m} not found in df_mks_mocap")

        # save to new CSV if needed
        df_jcp_mocap.to_csv(f"{base_path}/mocap_jcp/{subject_mocap}/{task}/joint_center_positions_w_offset_w_head.csv", index=False)