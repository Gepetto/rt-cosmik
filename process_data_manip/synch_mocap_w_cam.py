import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d


def downsample_mocap_to_lstm(task, no_trial, mocap_offset_sec=0.0):
    # === File paths ===
    base_path = f"/root/workspace/ros_ws/src/rt-cosmik/COSMIK_dataset/{no_trial}"
    base_path2 = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}"
    path_to_mocap = f"{base_path}/{task}/{task}_trajectories.csv"
    path_to_lstm = f"{base_path2}/cosmik_2cams/{task}/augmented_markers.csv"
    path_to_timestamps = f"{base_path2}/mouv/{task}/camera_0_timestamps.csv"
    
    # === Load data ===
    df_lstm = pd.read_csv(path_to_lstm)
    df_mocap = pd.read_csv(path_to_mocap)
    timestamps = pd.read_csv(path_to_timestamps)
    timestamps = pd.to_datetime(timestamps['timestamp'], errors="raise")
    
    # Convert to seconds from start
    timestamps_sec = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
    
    # Generate synthetic time for 100 Hz mocap
    num_mocap_samples = len(df_mocap)
    duration_sec = timestamps_sec[-1] + mocap_offset_sec
    synthetic_time_mocap = np.linspace(0 + mocap_offset_sec, duration_sec, num_mocap_samples)
    
    # Interpolate each column
    downsampled_columns = {}
    for col in df_mocap.columns:
        interp_func = interp1d(synthetic_time_mocap, df_mocap[col].values, kind='linear', fill_value="extrapolate")
        downsampled_columns[col] = interp_func(timestamps_sec)
    
    # === Save result ===
    df_downsampled = pd.DataFrame(downsampled_columns)
    output_path = f"{base_path2}/mocap/{task}/mocap_downsampled_to_40hz.csv"
    df_downsampled.to_csv(output_path, index=False)
    print(f"✅ Saved downsampled mocap aligned to 40Hz video: {output_path}")



if __name__ == "__main__":
    task_list = [ "bolting","bolting_sat","crouch","crouch_object","hitting","hitting_sat","jump","lifting",
    "lifting_fast","lower","overhead","overhead_front","robot_sanding","robot_welding",
    "sanding","sanding_sat","sit_to_stand","squat","static","upper","walk","walk_front",
    "welding","welding_sat"
]
    
    subjects = ["Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie","Guilhem","Kahina","Marie_M","Mathis",
     "Maxime_","Mohamed","Nicolas", "Zoe", "Herbert","Emmanuelle"
]

    for subject in subjects:
        for task in task_list:
            try:
                downsample_mocap_to_lstm(task, no_trial=subject)
            except FileNotFoundError:
                print(f"[SKIP] Missing data for subject={subject}, task={task}")