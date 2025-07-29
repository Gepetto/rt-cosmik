#plot mks meas vs mks model and calculate rmse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe

no_trial = "trial_2"
task = "trial_upper3"
csv_file1 = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_pose.csv"
csv_file2 = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_model.csv"


markers_to_plot = [
        'LBHD','RBHD','LFHD','RFHD',
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study'
    ]

# Load your data
df = pd.read_csv(csv_file1) #,nrows=4000)
df2 = pd.read_csv(csv_file2) #,nrows=4000)
mks_names = settings.marker_mocap_names
if 'marker_data' in df.columns:
    df = marker_data_to_dataframe(df, mks_names)

result_markers1, _ = read_mks_data(df)
result_markers2, _ = read_mks_data(df2)

def calculate_rmse_component(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    valid_mask = ~np.isnan(arr1) & ~np.isnan(arr2)
    return np.sqrt(np.mean((arr1[valid_mask] - arr2[valid_mask]) ** 2))

def plot_selected_markers(marker_data_1, marker_data_2, markers_to_plot=None):
    all_marker_names = set(marker_data_1[0].keys()).union(marker_data_2[0].keys())

    if markers_to_plot is None:
        markers_to_plot = all_marker_names

    for marker in markers_to_plot:
        if marker not in marker_data_1[0] and marker not in marker_data_2[0]:
            print(f"Marker '{marker}' not found in either dataset.")
            continue

        x_vals_1 = [frame.get(marker, [np.nan, np.nan, np.nan])[0] for frame in marker_data_1]
        y_vals_1 = [frame.get(marker, [np.nan, np.nan, np.nan])[1] for frame in marker_data_1]
        z_vals_1 = [frame.get(marker, [np.nan, np.nan, np.nan])[2] for frame in marker_data_1]

        x_vals_2 = [frame.get(marker, [np.nan, np.nan, np.nan])[0] for frame in marker_data_2]
        y_vals_2 = [frame.get(marker, [np.nan, np.nan, np.nan])[1] for frame in marker_data_2]
        z_vals_2 = [frame.get(marker, [np.nan, np.nan, np.nan])[2] for frame in marker_data_2]

        rmse_x = calculate_rmse_component(x_vals_1, x_vals_2)
        rmse_y = calculate_rmse_component(y_vals_1, y_vals_2)
        rmse_z = calculate_rmse_component(z_vals_1, z_vals_2)

        frames = np.arange(len(marker_data_1))

        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axs[0].plot(frames, x_vals_1, label='mks_meas', color='r')
        axs[0].plot(frames, x_vals_2, label='mks_model', color='b', linestyle='--')
        axs[0].set_title(f"{marker} - X | RMSE: {rmse_x} m")
        axs[0].set_ylabel("X")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(frames, y_vals_1, label='mks_meas', color='r')
        axs[1].plot(frames, y_vals_2, label='mks_model', color='b', linestyle='--')
        axs[1].set_title(f"{marker} - Y | RMSE: {rmse_y} m")
        axs[1].set_ylabel("Y")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(frames, z_vals_1, label='mks_meas', color='r')
        axs[2].plot(frames, z_vals_2, label='mks_model', color='b', linestyle='--')
        axs[2].set_title(f"{marker} - Z | RMSE: {rmse_z} m")
        axs[2].set_ylabel("Z")
        axs[2].set_xlabel("Frame")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()

plot_selected_markers(result_markers1, result_markers2, markers_to_plot=markers_to_plot)


# rmse_values = {}

# # Calculate RMSE for each marker
# for marker in marker_list:
#     columns = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
#     if all(col in data1.columns for col in columns) and all(col in data2.columns for col in columns):
#         # Calculate RMSE for x, y, z coordinates
#         rmse_x = calculate_rmse(data1, data2, columns[0])
#         rmse_y = calculate_rmse(data1, data2, columns[1])
#         rmse_z = calculate_rmse(data1, data2, columns[2])
        
#         # Store the total RMSE for the marker
#         rmse_values[marker] = np.mean([rmse_x, rmse_y, rmse_z])

# # Calculate average RMSE
# average_rmse = np.mean(list(rmse_values.values()))

# # Plot RMSE bar graph
# plt.figure(figsize=(12, 6))
# plt.bar(rmse_values.keys(), rmse_values.values(), color='skyblue', label='RMSE per Marker')
# plt.axhline(y=average_rmse, color='red', linestyle='--', linewidth=1.5, label=f'Average RMSE = {average_rmse:.4f}')
# plt.xticks(rotation=90, fontsize=8)
# plt.ylabel('RMSE')
# # plt.title('RMSE for Each Marker with Average')
# plt.legend()
# plt.tight_layout()
# plt.show()