#plot mks meas vs mks model and calculate rmse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def calculate_rmse(data1, data2, column):
    return np.sqrt(np.mean((data1[column] - data2[column]) ** 2))


csv_file1 = "mks_mocap/mks_mocap_test_2_corrected.csv" 
csv_file2 = "mks_mocap/mks_mocap_model_test_2.csv"  


# csv_file1 = "mks_lstm/augmented_markers_positions_by_rows_with_header.csv"  
# csv_file2 = "mks_lstm/mks_cosmik_model_test_2.csv"  
data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)

# List of markers to plot
marker_list = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study", "r_knee_study", "r_mknee_study",
    "r_ankle_study", "r_mankle_study", "r_toe_study", "r_5meta_study", "r_calc_study", "L_knee_study",
    "L_mknee_study", "L_ankle_study", "L_mankle_study", "L_toe_study", "L_calc_study", "L_5meta_study",
    "r_shoulder_study", "L_shoulder_study", "C7_study", "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study", "r_lelbow_study", "r_melbow_study", "r_lwrist_study",
    "r_mwrist_study", "L_lelbow_study", "L_melbow_study", "L_lwrist_study", "L_mwrist_study"
]

for marker in marker_list:
    # Extract columns for the marker (x, y, z)
    columns = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
    
    # Check if all columns exist in both datasets
    if all(col in data1.columns for col in columns) and all(col in data2.columns for col in columns):
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f"{marker}", fontsize=16)

        time = range(len(data1))  # Assuming time is the row index

        for i, coord in enumerate(['x', 'y', 'z']):
            col = f"{marker}_{coord}"
            # Calculate RMSE
            rmse = calculate_rmse(data1, data2, col)
            print(marker)
            print(rmse)
            
            # Plot the data
            axs[i].plot(time, data1[col], label="meas", color="r")
            axs[i].plot(time, data2[col], label="est", color="b", linestyle="--")
            axs[i].set_ylabel(f"{coord.upper()} Coordinate")
            axs[i].legend()
            
            # Add RMSE to the title of the subplot
            axs[i].set_title(f"RMSE = {rmse:.4f}")
        
        # Adjust layout
        axs[-1].set_xlabel("Time")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


rmse_values = {}

# Calculate RMSE for each marker
for marker in marker_list:
    columns = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
    if all(col in data1.columns for col in columns) and all(col in data2.columns for col in columns):
        # Calculate RMSE for x, y, z coordinates
        rmse_x = calculate_rmse(data1, data2, columns[0])
        rmse_y = calculate_rmse(data1, data2, columns[1])
        rmse_z = calculate_rmse(data1, data2, columns[2])
        
        # Store the total RMSE for the marker
        rmse_values[marker] = np.mean([rmse_x, rmse_y, rmse_z])

# Calculate average RMSE
average_rmse = np.mean(list(rmse_values.values()))

# Plot RMSE bar graph
plt.figure(figsize=(12, 6))
plt.bar(rmse_values.keys(), rmse_values.values(), color='skyblue', label='RMSE per Marker')
plt.axhline(y=average_rmse, color='red', linestyle='--', linewidth=1.5, label=f'Average RMSE = {average_rmse:.4f}')
plt.xticks(rotation=90, fontsize=8)
plt.ylabel('RMSE')
# plt.title('RMSE for Each Marker with Average')
plt.legend()
plt.tight_layout()
plt.show()