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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import argparse

# add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
parser.add_argument("--shuffle",
                    help="shuffle data or not",
                    dest="shuffle",
                    type=str2bool,
                    default=True)
parser.add_argument("--fine-tune",
                    help="Fine tune or retrain",
                    dest="fine_tune",
                    type=str2bool,
                    default=True)
parser.add_argument("--add-layer",
                    help="Add a final layer or not",
                    dest="add_layer",
                    type=str2bool,
                    default=True)
parser.add_argument("--mean-perso",
                    help="Use mean and std perso or not",
                    dest="mean_perso",
                    type=str2bool,
                    default=True)
opt = parser.parse_args()

# === Hyperparams ===
data_dir = opt.data_path
pretrained_dir = opt.pretrained_path
shuffle = opt.shuffle
fine_tune = opt.fine_tune
add_layer = opt.add_layer
mean_perso = opt.mean_perso
json_path      = os.path.join(pretrained_dir, "model.json")
weights_path   = os.path.join(pretrained_dir, "weights.h5")

test_size    = 0.2
random_state = 42
batch_size   = 64
epochs       = 100
patience     = 5
learning_rate= 6e-6
initializer = RandomNormal(mean=0.0, stddev=0.022)
weight_decay = 0.01

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

# Convert to numpy arrays
total_markers = default_mocap_mks_names
T = len(kpts_list)
mid_arr = np.zeros((T, 3), dtype=np.float32)
for i, frame in enumerate(kpts_list):
    mid_arr[i] = frame['midHip']

kpts_arr  = listdicts_to_array(kpts_list, kpts_input_lstm)
mocap_arr = listdicts_to_array(mocap_list, total_markers)

# === Load pretrained model ===
with open(json_path, 'r') as f:
    base = model_from_json(f.read())
base.load_weights(weights_path)

if fine_tune:
    # ❄️ Freeze tous les layers du modèle de base
    for layer in base.layers[:-1]:
        layer.trainable = False
    if add_layer:
        base.layers[-1].trainable = False

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

# ====== Set LSTM to output only last-step (last vector of the predicted window), then only with the markers of interest
if add_layer:
    projection = TimeDistributed(Dense(63), 
                                 kernel_initializer=initializer, 
                                 bias_initializer='zeros', 
                                 kernel_regularizer=l2(weight_decay), 
                                 name="dense_projection"
                                 )(base.output)
    # Final model
    model = Model(inputs=base.input, outputs=projection)
else:
    x = base.layers[-2].output
    new_output = TimeDistributed(Dense(63),
                                 kernel_initializer=initializer, 
                                 bias_initializer='zeros', 
                                 kernel_regularizer=l2(weight_decay), 
                                 name="replaced_output"
                                 )(x)
    model = Model(inputs=base.input, outputs=new_output)
model.summary()
model.compile(optimizer=Adam(learning_rate), loss='mse')

# Sauvegarde de l'architecture dans un fichier JSON
with open(os.path.join(pretrained_dir, f"model_finetuned_s{shuffle}_ft{fine_tune}_al{add_layer}_mp{mean_perso}.json"), "w") as f:
    f.write(model.to_json())

# indices for RHip/LHip in our kpts_input_lstm list
idx_RHip = kpts_input_lstm.index('RHip')
idx_LHip = kpts_input_lstm.index('LHip')

def data_generator(kpts_arr, mocap_arr, mid_arr, subject_heights, subject_weights,
                   chgt_subject_indexes, chgt_trial_indexes, seq_len, mks_of_interest_upper):
    
    start_subject = 0
    for ind_subject, end_subject in enumerate(chgt_subject_indexes):
        height = subject_heights[ind_subject]
        weight = subject_weights[ind_subject]

        subject_kpts = kpts_arr[start_subject:end_subject]
        subject_mocap = mocap_arr[start_subject:end_subject]
        subject_mid = mid_arr[start_subject:end_subject]

        start_trial = 0
        for _, end_trial in enumerate([i for i in chgt_trial_indexes if i <= end_subject]):
            for start in range(0, end_trial - start_trial - seq_len + 1):
                kbuf = subject_kpts[start:start+seq_len]
                # mbuf = subject_mocap[start+seq_len-1]
                ref = subject_mid[start:start+seq_len]

                norm  = kbuf - ref[:, None, :]
                norm2 = norm / height
                inp = norm2.reshape(seq_len, -1)
                inp = np.concatenate([
                    inp,
                    np.full((seq_len,1), height),
                    np.full((seq_len,1), weight)
                ], axis=1)

                # Apply normalization if files exist
                if mean_perso:
                    if os.path.isfile(os.path.join(pretrained_dir, "mean_perso.npy")):
                        mean = np.load(os.path.join(pretrained_dir, "mean_perso.npy"), allow_pickle=True)
                        inp -= mean
                    else :
                        raise Exception("Mean perso file does not exists.")
                    if os.path.isfile(os.path.join(pretrained_dir, "std_perso.npy")):
                        std = np.load(os.path.join(pretrained_dir, "std_perso.npy"), allow_pickle=True)
                        inp /= std
                    else :
                        raise Exception("Std perso file does not exists.")
                else:
                    if os.path.isfile(os.path.join(pretrained_dir, "mean.npy")):
                        mean = np.load(os.path.join(pretrained_dir, "mean.npy"), allow_pickle=True)
                        inp -= mean
                    else :
                        raise Exception("Mean perso file does not exists.")
                    if os.path.isfile(os.path.join(pretrained_dir, "std.npy")):
                        std = np.load(os.path.join(pretrained_dir, "std.npy"), allow_pickle=True)
                        inp /= std
                    else :
                        raise Exception("Std perso file does not exists.")

                sel = [default_mocap_mks_names.index(m) for m in mks_of_interest_upper]
                ybuf = subject_mocap[start:start+seq_len, sel, :]
                ybuf_norm = ybuf - ref[:, None, :]
                ybuf_norm2 = ybuf_norm / height
                out = ybuf_norm2.reshape(seq_len, -1)

                yield inp.astype(np.float32), out.astype(np.float32)
            
            start_trial = end_trial
        start_subject = end_subject

output_signature = (
    tf.TensorSpec(shape=(seq_len, kpts_arr.shape[2]*len(kpts_input_lstm)+2), dtype=tf.float32),
    tf.TensorSpec(shape=(seq_len, 3*len(mks_of_interest_lower)), dtype=tf.float32)
)

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(kpts_arr, mocap_arr, mid_arr, subjects_metadata["height"],
                           subjects_metadata["weight"], chgt_subject_indexes,
                           chgt_trial_indexes, seq_len, mks_of_interest_lower),
    output_signature=output_signature
)

total_samples = sum(1 for _ in dataset)
train_size = int((1 - test_size) * total_samples)

if shuffle:
    dataset = dataset.shuffle(buffer_size=total_samples, reshuffle_each_iteration=True)

train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# === Train ===
checkpoint = ModelCheckpoint(
    filepath=os.path.join(pretrained_dir, f"best_finetuned_weights_s{shuffle}_ft{fine_tune}_al{add_layer}_mp{mean_perso}.h5"),    
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,            # True si tu veux sauvegarder seulement les poids
    verbose=1
)
es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[es, checkpoint]
)


# Save fine-tuned weights
model.save_weights(os.path.join(pretrained_dir, f"weights_finetuned_s{shuffle}_ft{fine_tune}_al{add_layer}_mp{mean_perso}.h5"))
print("Fine-tuning complete. Saved to weights_finetuned.h5")
