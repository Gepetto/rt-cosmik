
import pandas as pd
import numpy as np


no_trial = "Mathis"
task = "squat"
path_to_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap_data/{task}_trajectories.csv"
path_to_lstm = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/augmented_markers_2.csv"
path_to_timestamps = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/camera_0_timestamps.csv"

# === Step 1: Load CSV files ===
timestamps = pd.read_csv(path_to_timestamps)  # timestamps for lstm
timestamps_datetime =  pd.to_datetime(timestamps['timestamp'], errors="raise")

df_40hz = pd.read_csv(path_to_lstm)
df_100hz = pd.read_csv(path_to_mocap)  # No timestamps

# === Step 2: Generate synthetic timestamps for 100Hz data ===
start_time = timestamps_datetime.iloc[0]
timestamps = (timestamps_datetime - start_time).dt.total_seconds()
num_samples_100hz = len(df_100hz)
synthetic_time_100hz = np.linspace(0, timestamps.iloc[-1], num_samples_100hz)

# === Step 3: Interpolate 100Hz data at 40Hz timestamps ===
data_100hz = df_100hz.values
downsampled_data = np.array([
    np.interp(timestamps, synthetic_time_100hz, data_100hz[:, i])
    for i in range(data_100hz.shape[1])
]).T  # Shape: (N, features)

# === Step 4: Save result (optional) ===
df_downsampled = pd.DataFrame(downsampled_data, columns=df_100hz.columns)
df_downsampled.to_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mocap_downsampled_to_40hz.csv", index=False)