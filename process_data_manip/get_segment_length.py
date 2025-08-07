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
from src.rtcosmik.human_model.model_utils import get_segment_length
import time
import csv
from src.rtcosmik.utils.read_write_utils import read_mks_data
base_path = "/root/workspace/ros_ws/src/rt-cosmik"

no_trial = "Mohamed"
task = "squat"
# === Subject physical info for LSTM ===
subject_mass =95.0
subject_height = 1.80
lstm_path = os.path.join(base_path, f"output/{no_trial}/cosmik_2cams/{task}/augmented_markers_2.csv")


lstm_mks = pd.read_csv(lstm_path)
# mocap_mks = pd.read_csv(os.path.join(rt_cosmik_path,'process_data_manip/mks_mocap_downsampled_33Hz.csv'))
path_for_segment_length = (os.path.join(rt_cosmik_path,'process_data_manip/segmtn_length_lstm.csv'))

start_sample=5
result_markers, lstm_dict = read_mks_data(lstm_mks, start_sample=start_sample)
lstm_dict = result_markers[start_sample]

# result_markers_mocap, mocap_dict = read_mks_data(mocap_mks, start_sample=start_sample)
# mocap_dict = result_markers_mocap[start_sample]

# get_segment_length(mocap_dict, path_for_segment_length)
get_segment_length(lstm_dict) 