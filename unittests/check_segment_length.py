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
from utils.model_w_mocap_utils import build_model_challenge, get_segment_length
from utils.ik_utils import RT_IK
from utils.viz_utils import place, Rquat
import time
import csv
from utils.read_write_utils import read_mks_data

lstm_mks = pd.read_csv(os.path.join(rt_cosmik_path,'process_data_manip/augmented_mks_interpolated_33hz.csv'))
mocap_mks = pd.read_csv(os.path.join(rt_cosmik_path,'process_data_manip/mks_mocap_downsampled_33Hz.csv'))
path_for_segment_length = (os.path.join(rt_cosmik_path,'process_data_manip/segmtn_length_lstm.csv'))

start_sample=5
result_markers, lstm_dict = read_mks_data(lstm_mks, start_sample=start_sample)
lstm_dict = result_markers[start_sample]

result_markers_mocap, mocap_dict = read_mks_data(mocap_mks, start_sample=start_sample)
mocap_dict = result_markers_mocap[start_sample]

# get_segment_length(mocap_dict, path_for_segment_length)
get_segment_length(lstm_dict, path_for_segment_length) 