from keras.models import Sequential, Model
from keras.layers import LSTM, TimeDistributed, Dense, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, default_mocap_mks_names


markers_lower = ['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 
                 'r_knee_study', 'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 
                 'r_toe_study', 'r_5meta_study', 'r_calc_study', 'L_knee_study', 'L_mknee_study', 
                 'L_ankle_study', 'L_mankle_study', 'L_toe_study', 'L_calc_study', 'L_5meta_study', 
                 'r_shoulder_study', 'L_shoulder_study', 'C7_study', 'r_thigh1_study', 'r_thigh2_study', 
                 'r_thigh3_study', 'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 
                 'r_sh2_study', 'r_sh3_study', 'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
markers_upper = ['r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 
                 'L_lelbow_study', 'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study']
marker_indices_lower = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20] #['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe']
marker_indices_upper = [18, 6, 5, 8, 7, 10, 9] #['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']

markers_mocap_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']
markers_mocap_names_extended = []
for marker_name in markers_mocap_names:
    markers_mocap_names_extended.append(f"{marker_name}_x")
    markers_mocap_names_extended.append(f"{marker_name}_y")
    markers_mocap_names_extended.append(f"{marker_name}_z")


# === Charger le CSV ===
df_inputs = pd.DataFrame()
df_gt = pd.DataFrame()
data_dir = "/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/Data_training_LSTM"
for subject in os.listdir(data_dir):
    subject_path = os.path.join(data_dir, subject)
    cosmik_2cams_path = os.path.join(subject_path, "cosmik_2cams")
    mocap_path = os.path.join(subject_path, "mocap")
    for trial in os.listdir(cosmik_2cams_path):
        if "mks_model_cosmik_2.csv" not in os.listdir(os.path.join(cosmik_2cams_path, trial)):
            print(f"Skipping {trial} in {subject} due to missing HPE data.")
            continue
        current_HPE_data_path = os.path.join(cosmik_2cams_path, trial, "mks_model_cosmik_2.csv")
        if "mks_data_gapfilled.csv" in os.listdir(os.path.join(mocap_path, trial)):
            current_mocap_data_path = os.path.join(mocap_path, trial, "mks_data_gapfilled.csv")
            current_df_gt = pd.read_csv(current_mocap_data_path)
            current_df_gt.drop(current_df_gt.columns[[0, 1]], axis=1, inplace=True)  # Suppression des deux premières colonnes (frame, subframe)
        elif "mks_data.csv" in os.listdir(os.path.join(mocap_path, trial)):
            current_mocap_data_path = os.path.join(mocap_path, trial, "mks_data.csv")
            current_df_gt = udp_csv_to_dataframe(current_mocap_data_path, markers_mocap_names, udp_type="raw")
        else:
            print(f"Skipping {trial} in {subject} due to missing mocap data.")
            continue
        current_df_inputs = pd.read_csv(current_HPE_data_path)
        current_df_inputs = current_df_inputs.iloc[:min(len(current_df_inputs), len(current_df_gt)),:]
        current_df_gt = current_df_gt.iloc[:min(len(current_df_inputs), len(current_df_gt)),:]
        current_df_gt.columns = markers_mocap_names_extended
        df_inputs = pd.concat([df_inputs, current_df_inputs], ignore_index=True)
        df_gt = pd.concat([df_gt, current_df_gt], ignore_index=True)

print(df_gt)

print("Shape of df_inputs:", df_inputs.shape)
print("Shape of df_gt:", df_gt.shape)
print("Columns in df_inputs:", df_inputs.columns)
print("Columns in df_gt:", df_gt.columns)

# === Conversion en tableau NumPy ===
data_inputs = df_inputs.to_numpy()
data_outputs = df_gt.to_numpy()

# === Séparation entrée / sortie ===
x = data_inputs[:, [18*3, 18*3+1, 18*3+2, 6*3, 6*3+1, 6*3+2, 5*3, 5*3+1, 5*3+2, 
                   8*3, 8*3+1, 8*3+2, 7*3, 7*3+1, 7*3+2, 10*3, 10*3+1, 10*3+2, 9*3, 9*3+1, 9*3+2]]
y = data_outputs[:, [8*3, 8*3+1, 8*3+2, 9*3, 9*3+1, 9*3+2, 10*3, 10*3+1, 10*3+2, 15*3, 
                 15*3+1, 15*3+2, 16*3, 16*3+1, 16*3+2, 18*3, 18*3+1, 18*3+2, 19*3, 19*3+1, 19*3+2,
                 24*3, 24*3+1, 24*3+2, 25*3, 25*3+1, 25*3+2, 27*3, 27*3+1, 27*3+2, 28*3, 28*3+1, 28*3+2]]


# === Reshape en (batch, time, features) ===
x = x.reshape((-1, 21))
y = y.reshape((-1, 33))

print("Shape of x:", x.shape)  # (nb_samples, timesteps, input_dim)
print("Shape of y:", y.shape)  # (nb_samples, timesteps, output_dim)


# === Split train / val ===
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.reshape((-1, 1, 21))  # (nb_samples, timesteps, input_dim)
x_val = x_val.reshape((-1, 1, 21))      # (nb_samples, timesteps, input_dim)
y_train = y_train.reshape((-1, 1, 33))  # (nb_samples, timesteps, output_dim)
y_val = y_val.reshape((-1, 1, 33))      # (nb_samples, timesteps, output_dim)

# === Hyperparamètres ===
input_dim = 21       # nb de features en entrée
output_dim = 24      # nb de coordonnées/keypoints à prédire
timesteps = None     # ou fixe si besoin

# # === Reconstruit le modèle jusqu’à l’avant-dernière couche ===
# inputs = Input(shape=(timesteps, input_dim), name="input")
# x = LSTM(128, return_sequences=True)(inputs)
# x = LSTM(128, return_sequences=True)(x)
# x = LSTM(128, return_sequences=True)(x)
# x = LSTM(128, return_sequences=True)(x)

# base_model = Model(inputs, x, name="base_model")
# base_model.load_weights("/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_upper/weights.h5")  # Charger les poids du modèle pré-entraîné

# # === Remplace la sortie par une nouvelle couche ===
# new_outputs = TimeDistributed(Dense(output_dim, activation="linear"), name="fine_tune_output")(base_model.output)

# Chemin vers votre fichier JSON
json_path = "/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_upper/model.json"

# Lire le contenu du fichier
with open(json_path, 'r') as file:
    model_config_json = file.read()

    base_model = model_from_json(model_config_json)

base_model.load_weights("/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_upper/weights.h5")

# === Remplace la sortie par une nouvelle couche ===
new_outputs = TimeDistributed(Dense(output_dim, activation="linear"), name="fine_tune_output")(base_model.output)

fine_tune_model = Model(base_model.input, outputs=new_outputs)
for layer in fine_tune_model.layers[:-1]:
    layer.trainable = False
fine_tune_model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # ou autre métrique si classification
fine_tune_model.summary()


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = fine_tune_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

fine_tune_model.save("/mnt/c/Users/nicol/Desktop/Travail/LAAS/SFE Gepetto/rt-cosmik/src/rtcosmik/augmenter/augmentation_model/LSTM/fine_tuned_upper_lstm_model.h5")

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# MAE (ou autre)
if 'mae' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

plt.tight_layout()
plt.show()
