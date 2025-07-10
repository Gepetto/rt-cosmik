#!/usr/bin/env python3
# fine_tune_lstm.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# Add the src folder to sys.path so that viewer modules can be found.cd ..
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.rtcosmik.utils.read_write_utils import *

# === Hyperparams ===
data_dir           = "./data/lstm_training"
pretrained_dir     = "./models/LSTM/v0.3_lower"  # <-- set to your pretrained JSON+H5 folder
json_path          = os.path.join(pretrained_dir, "model.json")
weights_path       = os.path.join(pretrained_dir, "weights.h5")

test_size          = 0.2
random_state       = 42
batch_size         = 32
epochs             = 100
patience           = 10
learning_rate      = 1e-3  # you can tune this

# === Paths ===
kpts_csv    = os.path.join(data_dir, "3d_keypoints_filtered_4.csv")
mocap_csv   = os.path.join(data_dir, "mks_data_gapfilled.csv")

# === Markers and keypoints names ===
kpts_input_lstm = ['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 
                    'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
mks_of_interest_lower = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study' ,'C7_study','r_shoulder_study','L_shoulder_study',
                        'L_knee_study','L_mknee_study', 'L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study',
                        'L_toe_study', 'r_knee_study','r_mknee_study',
                        'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study']
response_markers_lower =['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 
                        'r_mankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 
                        'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 
                        'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 
                        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 
                        'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']

mocap_mks = udp_csv_to_dataframe(mocap_csv, default_mocap_mks_names) #float
mocap_mks, _ = read_mks_data(mocap_mks)
kpts = pd.read_csv(kpts_csv)
kpts, _ = read_mks_data(kpts) #is a list of dictionnary

print(len(mocap_mks), "  ", len(mocap_mks[0]))
print(len(kpts), "  ", len(kpts[0]))

