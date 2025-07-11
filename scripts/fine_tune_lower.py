#!/usr/bin/env python3
# fine_tune_lstm.py

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

# add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names

# === Hyperparams ===
data_dir       = "./data/lstm_training"
pretrained_dir = "./models/LSTM/v0.3_lower"
json_path      = os.path.join(pretrained_dir, "model.json")
weights_path   = os.path.join(pretrained_dir, "weights.h5")

test_size    = 0.2
random_state = 42
batch_size   = 32
epochs       = 100
patience     = 10
learning_rate= 1e-3

# === Marker / keypoint names ===
kpts_input_lstm = [
    'Neck','RShoulder','LShoulder','RHip','LHip',
    'RKnee','LKnee','RAnkle','LAnkle','RHeel',
    'LHeel','RSmallToe','LSmallToe','RBigToe','LBigToe'
]

response_markers_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study',
    'r_thigh1_study','r_thigh2_study','r_thigh3_study',
    'L_thigh1_study','L_thigh2_study','L_thigh3_study',
    'r_sh1_study','r_sh2_study','r_sh3_study',
    'L_sh1_study','L_sh2_study','L_sh3_study',
    'RHJC_study','LHJC_study'
]

mks_of_interest_lower = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
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
kpts_csv  = os.path.join(data_dir, "3d_keypoints_filtered_4.csv")
mocap_csv = os.path.join(data_dir, "mks_data_gapfilled.csv")

mocap_df     = udp_csv_to_dataframe(mocap_csv, default_mocap_mks_names)
mocap_list,_ = read_mks_data(mocap_df)
kpts_df      = pd.read_csv(kpts_csv)
kpts_list,_  = read_mks_data(kpts_df)

# Convert to numpy arrays
total_markers = default_mocap_mks_names
T = len(kpts_list)

kpts_arr  = listdicts_to_array(kpts_list, kpts_input_lstm)
mocap_arr = listdicts_to_array(mocap_list, total_markers)

# === Load pretrained model ===
with open(json_path, 'r') as f:
    base = model_from_json(f.read())
base.load_weights(weights_path)

# Set sequence length (the LSTM was pre-trained to work with any sequence)
seq_len = 30
# total output dims (unused directly)
total_out_dim = base.output_shape[-1]
print(f"Using seq_len={seq_len}, total_out_dim={total_out_dim}")

# ====== Get the indices of the markers of interest in the LSTM full output ====== #
marker_idx = {m:i for i,m in enumerate(response_markers_lower)} #dictionnary of output lstm mks and their indices in the output vector
feat_indices = []#The indices of the features of interest
for m in mks_of_interest_lower:
    idx = marker_idx[m]
    feat_indices += [idx*3 + d for d in (0,1,2)]

# ====== Set LSTM to output only last-step (last vector of the predicted window), then only the markers of interest
last_step = Lambda(lambda x: x[:, -1, :], name="last_step")(base.output) #A lambda layer that outputs laststep only
lower_out = Lambda(lambda x: tf.gather(x, feat_indices, axis=1), #A lambda layer that outputs markers of interest only
                   name="lower_body")(last_step)
model = Model(inputs=base.input, outputs=lower_out)
model.compile(optimizer=Adam(learning_rate), loss='mse')

# indices for RHip/LHip in our kpts_input_lstm list
idx_RHip = kpts_input_lstm.index('RHip')
idx_LHip = kpts_input_lstm.index('LHip')

# === Prepare X, y ===
X, y = [], []
for start in range(0, T - seq_len + 1):
    kbuf = kpts_arr[start:start+seq_len]            # (seq_len, 15, 3)
    mbuf = mocap_arr[start+seq_len-1]               # (M, 3)

    # compute mid-hip as average of RHip and LHip
    ref = (kbuf[:, idx_RHip, :] + kbuf[:, idx_LHip, :]) / 2

    # center all keypoints by ref
    norm = kbuf - ref[:, None, :]
    norm2 = norm / 1.75  # subject_height (TODO: load per-subject)

    # flatten sequence + features
    inp = norm2.reshape(seq_len, -1)
    # append height & mass features
    inp = np.concatenate([
        inp,
        np.full((seq_len,1), 1.75),  # height
        np.full((seq_len,1), 70.0)    # mass
    ], axis=1)

    # apply pretrained mean/std
    mean_p = os.path.join(pretrained_dir, "mean.npy")
    std_p  = os.path.join(pretrained_dir, "std.npy")
    if os.path.isfile(mean_p):
        inp -= np.load(mean_p)
    if os.path.isfile(std_p):
        inp /= np.load(std_p)

    X.append(inp)
    # build target for markers_of_interest
    sel = [total_markers.index(m) for m in mks_of_interest_lower]
    y.append(mbuf[sel].reshape(-1))

X = np.stack(X)
y = np.stack(y)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size,
    random_state=random_state, shuffle=True
)

# === Train ===
es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[es]
)

# Save fine-tuned weights
model.save_weights(os.path.join(pretrained_dir, "weights_finetuned.h5"))
print("Fine-tuning complete. Saved to weights_finetuned.h5")
