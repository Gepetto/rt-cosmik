#!/usr/bin/env python3
# fine_tune_lstm_upper.py

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names

# === Hyperparams ===
data_dir       = sys.argv[1]  # Path to the directory containing subject data
pretrained_dir = sys.argv[2]  # Path to the pretrained model directory
json_path      = os.path.join(pretrained_dir, "model.json")
weights_path   = os.path.join(pretrained_dir, "weights.h5")

test_size     = 0.2
random_state  = 42
batch_size    = 64
epochs        = 10
patience      = 2
learning_rate = 6e-6

# === Marker / keypoint names (upper limb) ===
kpts_input_lstm = [
    'Neck', 'RShoulder', 'LShoulder',
    'RElbow', 'LElbow', 'RWrist', 'LWrist'
]

response_markers_upper = [
    'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
]

mks_of_interest_upper = response_markers_upper.copy()

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
        if "3d_keypoints_filtered_2_cleaned.csv" not in os.listdir(os.path.join(cosmik_2cams_path, trial)):
            print(f"Skipping {trial} in {subject} due to missing HPE data.")
            continue
        current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2_cleaned.csv")
        current_df_inputs = pd.read_csv(current_HPE_data_path)

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

# Convert to numpy arrays
kpts_arr  = listdicts_to_array(kpts_list, kpts_input_lstm)
mocap_arr = listdicts_to_array(mocap_list, default_mocap_mks_names)

# === Load pretrained model ===
with open(json_path, 'r') as f:
    base = model_from_json(f.read())
base.load_weights(weights_path)

# fixed sequence length used in original training
seq_len = 30
# total output dims (unused directly)
total_out_dim = base.output_shape[-1]
print(f"Using seq_len={seq_len}, total_out_dim={total_out_dim}")

# ===== Build fine-tuning model =====
marker_idx = {m:i for i,m in enumerate(response_markers_upper)}
feat_indices = []
for m in mks_of_interest_upper:
    idx = marker_idx[m]
    feat_indices += [idx*3 + d for d in (0,1,2)]

last_step = Lambda(lambda x: x[:, -1, :], name="last_step")(base.output)
upper_out = Lambda(lambda x: tf.gather(x, feat_indices, axis=1), name="upper_body")(last_step)
model     = Model(inputs=base.input, outputs=upper_out)
model.summary()
model.compile(optimizer=Adam(learning_rate), loss='mse')

# === Prepare X, y for fine-tuning ===
X, y = [], []

for ind_subject, chgt_subject_index in enumerate(chgt_subject_indexes):
    for ind_trial, chgt_trial_index in enumerate(chgt_trial_indexes):
        for start in range(0, chgt_trial_index - seq_len + 1):
            kbuf = kpts_arr[start:start+seq_len]    # (seq_len,7,3)
            mbuf = mocap_arr[start+seq_len-1]       # (M,3)

            # reference = mid-hip
            ref = mid_arr[start:start+seq_len]      # (seq_len,3)

            # center all keypoints by ref
            norm  = kbuf - ref[:, None, :]
            norm2 = norm / subjects_metadata["height"][ind_subject]

            # flatten + append height/mass
            inp = norm2.reshape(seq_len, -1)
            inp = np.concatenate([
                inp,
                np.full((seq_len,1), subjects_metadata["height"][ind_subject]),
                np.full((seq_len,1), subjects_metadata["weight"][ind_subject])
            ], axis=1)

            # apply pretrained mean/std
            mean_p = os.path.join(pretrained_dir, "mean.npy")
            std_p  = os.path.join(pretrained_dir, "std.npy")
            if os.path.isfile(mean_p): inp -= np.load(mean_p)
            if os.path.isfile(std_p):  inp /= np.load(std_p)

            X.append(inp)
            sel = [default_mocap_mks_names.index(m) for m in mks_of_interest_upper]
            y.append(mbuf[sel].reshape(-1))

X = np.stack(X)
y = np.stack(y)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size,
    random_state=random_state, shuffle=True
)

# === Train ===
checkpoint = ModelCheckpoint(
    filepath=os.path.join(pretrained_dir, "best_finetuned_weights.h5"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,          
    verbose=1
)
es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[es, checkpoint]
)

# Sauvegarde de l'architecture dans un fichier JSON
with open(os.path.join(pretrained_dir, "model_finetuned.json"), "w") as f:
    f.write(model.to_json())

# Save fine-tuned weights
model.save_weights(os.path.join(pretrained_dir, "weights_finetuned.h5"))
print("Fine-tuning complete. Saved to weights_finetuned.h5")