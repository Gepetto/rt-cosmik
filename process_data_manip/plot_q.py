#plot q_cosmik and q_mocap to check if i have same pattern
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint
no_trial = "trial_2"
task = "trial_lower"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mocap_on_cosmik_frames.csv"

path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_cosmik_ipopt.csv" #real time : 9.7deg bas du corps, 11deg haut du corps.
#offline lower 7.4deg , upper 8deg

dofs  = settings.joint_angles_names

upper_dof = ['Lumbar_flex_ext', 'Lumbar_int_ext_rot',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
                          'Relbow_flex_ext', 'Relbow_pron_supi', 'Lshoulder_flex_ext',
                          'Lshoulder_abd_add', 'Lshoulder_int_ext_rot', 'Lelbow_flex_ext',
                          'Lelbow_pron_supi']
                          
lower_dof=['Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot', 'Lknee_flex_ext', 'Lankle_flex_ext']

start_sample = 0
# q_cosmik= read_joint_angles_wholebody(path_cosmik, start_sample)
# q_mocap = read_joint_angles_wholebody(path_mocap, start_sample)

q_cosmik= read_specific_joint(path_cosmik,lower_dof, start_sample)
q_mocap = read_specific_joint(path_mocap,lower_dof, start_sample)

rmse_list = []
# Plot one figure per joint
for i, name in enumerate(lower_dof):
    rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
    rmse = rmse * (180 / np.pi)
    rmse_list.append(rmse)
    plt.figure()
    plt.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
    plt.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
    plt.title(f"{name} (RMSE: {rmse:.4f})")
    plt.xlabel("samples")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


average_rmse = np.mean(rmse_list)
print(f"\nAverage RMSE across all joints: {average_rmse:.4f} deg")

# groups = [
#     ["middle_lumbar_Z", "middle_lumbar_Y"],
#     ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"],
#     ["left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y"],
#     ["left_elbow_Z", "left_elbow_Y"],
#     ["right_elbow_Z", "right_elbow_Y"],
#     ["right_hip_Z", "right_hip_X", "right_hip_Y"],
#     ["left_hip_Z", "left_hip_X", "left_hip_Y"],
#     ["right_knee_Z"],
#     ["left_knee_Z"],
#     ["right_ankle_Z"],
#     ["left_ankle_Z"]
# ]

# # Map column indices for each DOF
# dof_indices = {dof: i for i, dof in enumerate(dofs)}

# # Plot each group in a separate figure with subplots and calculate RMSE
# for group in groups:
#     num_subplots = len(group)
#     fig, axs = plt.subplots(num_subplots, 1, figsize=(8, num_subplots * 4))
    
#     if num_subplots == 1:  # Ensure axs is always iterable
#         axs = [axs]
    
#     for ax, joint in zip(axs, group):
#         if joint in dof_indices:
#             idx = dof_indices[joint]
            
#             # Mocap and Cosmik data
#             mocap_data = y.values[:, idx]
#             cosmik_data = y2.values[:, idx]
            
#             # Calculate RMSE
#             rmse = np.sqrt(np.mean((mocap_data - cosmik_data) ** 2))
#             rmse = rmse * (180 / np.pi)
            
#             # Plot data from both CSVs
#             ax.plot(mocap_data, label=f'mocap', linestyle='-', linewidth=2, color='blue')
#             ax.plot(cosmik_data, label=f'cosmik', linestyle='--', linewidth=2, color='orange')
            
#             # Add titles, labels, and RMSE to the title
#             ax.set_title(f"{joint} (RMSE: {rmse:.4f})")
#             ax.legend()
#             ax.grid()
    
# plt.tight_layout()
# plt.show()