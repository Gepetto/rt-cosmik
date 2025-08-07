#!/usr/bin/env python3
# fine_tune_lstm_upper.py

import os
import sys
import numpy as np
import pandas as pd
import argparse

# add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names

# === Args ===
parser = argparse.ArgumentParser(description='LSTM retraining/finetuning arguments')
parser.add_argument('--data-path',
                        help='data path',
                        dest='data_path',
                        default='',
                        type=str)
parser.add_argument('--pretrained-path',
                        help='pretrained path',
                        dest='pretrained_path',
                        default='',
                        type=str)
opt = parser.parse_args()

# === Hyperparams ===
data_dir = opt.data_path
pretrained_dir = opt.pretrained_path

# === Marker / keypoint names (upper limb) ===
kpts_input_upper_lstm = [
    'Neck', 'RShoulder', 'LShoulder',
    'RElbow', 'LElbow', 'RWrist', 'LWrist'
]
kpts_input_upper_lstm_extended = list(np.array([[f"{m}_x", f"{m}_y", f"{m}_z"] for m in kpts_input_upper_lstm]).flatten())

kpts_input_lower_lstm = [
    'Neck','RShoulder','LShoulder','RHip','LHip',
    'RKnee','LKnee','RAnkle','LAnkle','RHeel',
    'LHeel','RSmallToe','LSmallToe','RBigToe','LBigToe'
]
kpts_input_lower_lstm_extended = list(np.array([[f"{m}_x", f"{m}_y", f"{m}_z"] for m in kpts_input_lower_lstm]).flatten())

excluded_trials = [
    "welding_sat",
    "sanding_sat",
    "hitting_sat",
    "bolting_sat",
    "crouch_object",
    "robot_sanding",
    "robot_welding",
]

# === Utility converters ===
def listdicts_to_array(ld, names):
    """Builds array [T, len(names), 3] from list-of-dicts using given keys."""
    T = len(ld)
    P = len(names)
    arr = np.zeros((T, P, 3), dtype=np.float32)
    for i, frame in enumerate(ld):
        for j, key in enumerate(names):
            arr[i, j, :] = frame[key]
    return arr

# === Load data ===
df_inputs = pd.DataFrame()
df_gt = pd.DataFrame()

subjects_metadata = {}
subjects_metadata["name"] = []
subjects_metadata["height"] = []
subjects_metadata["weight"] = []
chgt_subject_indexes = []
chgt_trial_indexes = []
for subject in os.listdir(data_dir):
    subject_path = os.path.join(data_dir, subject)

    metadata_path = os.path.join(subject_path, "infos.txt")
    with open(metadata_path, 'r') as f:
        metadata = f.readlines()
    subjects_metadata["name"].append(subject)
    subjects_metadata["height"].append(float(metadata[0].strip().split(":")[1]))
    subjects_metadata["weight"].append(float(metadata[1].strip().split(":")[1]))

    cosmik_2cams_path = os.path.join(subject_path, "cosmik_2cams")
    mocap_path = os.path.join(subject_path, "mocap")

    for trial in os.listdir(cosmik_2cams_path):
        if any(keyword in trial for keyword in excluded_trials):
            print(f"Skipping {trial} in {subject} due to HPE bug.")
            continue
        
        if "3d_keypoints_filtered_2.csv" in os.listdir(os.path.join(cosmik_2cams_path, trial)):
            current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2.csv")
            current_df_inputs = pd.read_csv(current_HPE_data_path)
        elif "3d_keypoints_filtered_2_cleaned.csv" in os.listdir(os.path.join(cosmik_2cams_path, trial)):
            current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2_cleaned.csv")
            current_df_inputs = pd.read_csv(current_HPE_data_path)
        else:
            print(f"Skipping {trial} in {subject} due to missing HPE data.")
            continue

        if "mks_data_cleaned.csv" in os.listdir(os.path.join(mocap_path, trial)):
            current_mocap_data_path = os.path.join(mocap_path, trial, "mks_data_cleaned.csv")
            current_df_gt = pd.read_csv(current_mocap_data_path)
        elif "mks_data_gapfilled.csv" in os.listdir(os.path.join(mocap_path, trial)):
            current_mocap_data_path = os.path.join(mocap_path, trial, "mks_data_gapfilled.csv")
            current_df_gt = udp_csv_to_dataframe(current_mocap_data_path, default_mocap_mks_names, udp_type="gapfilled")
        elif "mks_data.csv" in os.listdir(os.path.join(mocap_path, trial)):
            current_mocap_data_path = os.path.join(mocap_path, trial, "mks_data.csv")
            current_df_gt = udp_csv_to_dataframe(current_mocap_data_path, default_mocap_mks_names, udp_type="raw")
        else:
            print(f"Skipping {trial} in {subject} due to missing mocap data.")
            continue

        # équilibrage des longueurs des datas (si une frame en plus dans l'un ou l'autre)
        current_df_inputs = current_df_inputs.iloc[:min(len(current_df_inputs), len(current_df_gt)),:]
        current_df_gt = current_df_gt.iloc[:min(len(current_df_inputs), len(current_df_gt)),:]

        df_inputs = pd.concat([df_inputs, current_df_inputs], ignore_index=True)
        df_gt = pd.concat([df_gt, current_df_gt], ignore_index=True)

        chgt_trial_indexes.append(len(df_inputs))
    
    chgt_subject_indexes.append(len(df_inputs))

mocap_df = df_gt.copy()
mocap_list, _ = read_mks_data(mocap_df)
kpts_df = df_inputs.copy()
kpts_list, _ = read_mks_data(kpts_df)

# build mid-hip reference array
T = len(kpts_list)
mid_arr = np.zeros((T, 3), dtype=np.float32)
for i, frame in enumerate(kpts_list):
    mid_arr[i] = frame['midHip']

for ind_col, col in enumerate(df_inputs.columns):
    if col.endswith("_x"):
        df_inputs.iloc[:, ind_col] -= mid_arr[:, 0]
    elif col.endswith("_y"):
        df_inputs.iloc[:, ind_col] -= mid_arr[:, 1]
    elif col.endswith("_z"):
        df_inputs.iloc[:, ind_col] -= mid_arr[:, 2]
old_chgt_index = 0
for ind_subject, chgt_index in enumerate(chgt_subject_indexes):
    df_inputs.iloc[old_chgt_index:chgt_index, :] /= subjects_metadata["height"][ind_subject]
    old_chgt_index = chgt_index
mean_inputs_upper = df_inputs[kpts_input_upper_lstm_extended].mean().values
std_inputs_upper = df_inputs[kpts_input_upper_lstm_extended].std().values
mean_inputs_lower = df_inputs[kpts_input_lower_lstm_extended].mean().values
std_inputs_lower = df_inputs[kpts_input_lower_lstm_extended].std().values
height_mean = np.mean(subjects_metadata["height"])
height_std = np.std(subjects_metadata["height"])
weight_mean = np.mean(subjects_metadata["weight"])
weight_std = np.std(subjects_metadata["weight"])
mean_inputs_upper = np.concatenate((mean_inputs_upper, [height_mean, weight_mean]))
std_inputs_upper = np.concatenate((std_inputs_upper, [height_std, weight_std]))
mean_inputs_lower = np.concatenate((mean_inputs_lower, [height_mean, weight_mean]))
std_inputs_lower = np.concatenate((std_inputs_lower, [height_std, weight_std]))
print("Mean inputs upper:", mean_inputs_upper)
print("Std inputs upper:", std_inputs_upper)
print("Mean inputs lower:", mean_inputs_lower)
print("Std inputs lower:", std_inputs_lower)
np.save(os.path.join(pretrained_dir, "v0.3_upper", "mean_perso.npy"), mean_inputs_upper)
np.save(os.path.join(pretrained_dir, "v0.3_upper", "std_perso.npy"), std_inputs_upper)
np.save(os.path.join(pretrained_dir, "v0.3_lower", "mean_perso.npy"), mean_inputs_lower)
np.save(os.path.join(pretrained_dir, "v0.3_lower", "std_perso.npy"), std_inputs_lower)