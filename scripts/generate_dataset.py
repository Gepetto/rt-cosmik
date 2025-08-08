#!/usr/bin/env python3
# fine_tune_lstm_upper.py

import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names

# === Args ===
parser = argparse.ArgumentParser(description='LSTM retraining/finetuning dataset generation')
parser.add_argument('--data-path',
                        help='data path',
                        dest='data_path',
                        default='',
                        type=str)
parser.add_argument('--output-path',
                        help='output path',
                        dest='output_path',
                        default='',
                        type=str)
parser.add_argument('--body-part',
                        help='body part',
                        dest='body_part',
                        default='',
                        type=str)
parser.add_argument('--use-mocap',
                        help='use mocap or hpe',
                        dest='use_mocap',
                        default='',
                        type=str)
opt = parser.parse_args()

data_dir = opt.data_path
output_dir = opt.output_path
body_part = opt.body_part
use_mocap = opt.use_mocap

test_size     = 0.2
random_state  = 42
seq_len       = 30

# === Utils ===
if body_part == "upper":
    kpts_input_lstm = [
    'Neck', 'RShoulder', 'LShoulder',
    'RElbow', 'LElbow', 'RWrist', 'LWrist'
    ]
    mks_of_interest = [
    'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
    ]
elif body_part == "lower":
    kpts_input_lstm = [
    'Neck','RShoulder','LShoulder','RHip','LHip',
    'RKnee','LKnee','RAnkle','LAnkle','RHeel',
    'LHeel','RSmallToe','LSmallToe','RBigToe','LBigToe'
    ]
    mks_of_interest = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
    ]
else:
    raise Exception("Body part not supported. Please select upper or lower.")

excluded_trials = [
    "welding_sat",
    "sanding_sat",
    "hitting_sat",
    "bolting_sat",
    "crouch_object",
    "robot_sanding",
    "robot_welding",
]

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
    print("subject :", subject)
    for trial in os.listdir(cosmik_2cams_path):
        # Skip trials with HPE bug
        if any(keyword in trial for keyword in excluded_trials):
            print(f"Skipping {trial} in {subject} due to HPE bug.")
            continue
        
        print("trial :", trial)
        # Wether we use cleaned HPE data or raw HPE data
        if use_mocap == "T":
            if "joint_center_positions.csv" in os.listdir(os.path.join(mocap_path, trial)):
                current_input_data_path = os.path.join(mocap_path, trial, "joint_center_positions.csv")
                current_df_inputs = pd.read_csv(current_input_data_path)
            else:
                print(f"Skipping {trial} in {subject} due to missing JCP mocap data.")
                continue
        elif use_mocap == "F":
            if "3d_keypoints_filtered_2.csv" in os.listdir(os.path.join(cosmik_2cams_path, trial)):
                current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2.csv")
                current_df_inputs = pd.read_csv(current_HPE_data_path)
            elif "3d_keypoints_filtered_2_cleaned.csv" in os.listdir(os.path.join(cosmik_2cams_path, trial)):
                current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2_cleaned.csv")
                current_df_inputs = pd.read_csv(current_HPE_data_path)
            else:
                print(f"Skipping {trial} in {subject} due to missing HPE data.")
                continue
        else:
            raise Exception("Please specify --use-mocap argument as T or F.")

        # Wether we use cleaned mocap data or raw mocap data
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

        # Concatenate HPE and mocap data to previous
        df_inputs = pd.concat([df_inputs, current_df_inputs], ignore_index=True)
        df_gt = pd.concat([df_gt, current_df_gt], ignore_index=True)

        # Save trial changement indexes
        chgt_trial_indexes.append(len(df_inputs))
    
    # Save subject changement indexes
    chgt_subject_indexes.append(len(df_inputs))

# Convert to lists arrays
mocap_df = df_gt.copy()
mocap_list, _ = read_mks_data(mocap_df)
kpts_df = df_inputs.copy()
kpts_list, _ = read_mks_data(kpts_df)


# build mid-hip reference array
T = len(kpts_list)
mid_arr = np.zeros((T, 3), dtype=np.float32)
for i, sample in enumerate(kpts_list):
    mid_arr[i] = sample['midHip']

# Convert to numpy arrays
kpts_arr  = listdicts_to_array(kpts_list, kpts_input_lstm)
mocap_arr = listdicts_to_array(mocap_list, default_mocap_mks_names)

# === generate dataset ===
def data_generator(kpts_arr, mocap_arr, mid_arr, subject_heights, subject_weights,
                   chgt_subject_indexes, chgt_trial_indexes, seq_len, mks_of_interest):
    
    start_subject = 0
    start_trial = 0
    for ind_subject, end_subject in enumerate(chgt_subject_indexes):
        height = subject_heights[ind_subject]
        weight = subject_weights[ind_subject]

        subject_kpts = kpts_arr[start_subject:end_subject]
        subject_mocap = mocap_arr[start_subject:end_subject]
        subject_mid = mid_arr[start_subject:end_subject]

        for end_trial in [i for i in chgt_trial_indexes if i <= end_subject and i > start_subject]:
            for start in range(0, end_trial - start_trial - seq_len + 1):
                kbuf = subject_kpts[start:start+seq_len]
                ref = subject_mid[start:start+seq_len]

                inp = kbuf - ref[:, None, :]
                inp = inp / height
                inp = inp.reshape(seq_len, -1)
                inp = np.concatenate([
                    inp,
                    np.full((seq_len,1), height),
                    np.full((seq_len,1), weight)
                ], axis=1)

                sel = [default_mocap_mks_names.index(m) for m in mks_of_interest]
                ybuf = subject_mocap[start:start+seq_len, sel, :]
                out = ybuf - ref[:, None, :]
                out = out / height
                out = out.reshape(seq_len, -1)

                yield inp.astype(np.float64), out.astype(np.float64)
        
            start_trial = end_trial
        start_subject = end_subject

# Collecte
X, Y = [], []
for x, y in data_generator(kpts_arr, mocap_arr, mid_arr, subjects_metadata["height"],
                           subjects_metadata["weight"], chgt_subject_indexes,
                           chgt_trial_indexes, seq_len, mks_of_interest):
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=42)
print("X_train :", X_train.shape)
print("Y_train :", Y_train.shape)
print("X_val :", X_val.shape)
print("Y_val :", Y_val.shape)

# Calculate mean and std
mean_train = np.mean(X_train.astype(np.float64), axis=(0, 1))
std_train = np.std(X_train.astype(np.float64), axis=(0, 1))
print("Mean train :", mean_train)
print("Std train :", std_train)

# === Normalize data ===
X_train = (X_train - mean_train) / std_train
X_val = (X_val - mean_train) / std_train

# Save
os.makedirs(os.path.join(output_dir, body_part, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, body_part, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, body_part, "stats"), exist_ok=True)
np.save(os.path.join(output_dir, body_part, "train", f"X_train_m{use_mocap}.npy"), X_train)
np.save(os.path.join(output_dir, body_part, "train", "Y_train.npy"), Y_train)
np.save(os.path.join(output_dir, body_part, "val", f"X_val_m{use_mocap}.npy"), X_val)
np.save(os.path.join(output_dir, body_part, "val", "Y_val.npy"), Y_val)
np.save(os.path.join(output_dir, body_part, "stats", f"mean_train_m{use_mocap}.npy"), mean_train)
np.save(os.path.join(output_dir, body_part, "stats", f"std_train_m{use_mocap}.npy"), std_train)
