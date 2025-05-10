#plot q_cosmik and q_mocap to check if i have same pattern
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint

no_trial = "trial3"
task = "overhead"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_mocap_ipopt_filtred.csv"

path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_cosmik_ipopt_filtred.csv" 

dofs  = settings.joint_angles_names

upper_dof = ['Lumbar_flex_ext', 'Lumbar_int_ext_rot',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
                          'Lshoulder_flex_ext',
                          'Lshoulder_abd_add', 'Lshoulder_int_ext_rot',
                          'Relbow_flex_ext', 'Relbow_pron_supi', 'Lelbow_flex_ext',
                          'Lelbow_pron_supi']
                          
lower_dof=['Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Lknee_flex_ext', 'Lankle_flex_ext']

up_low = "upper"
if up_low =='upper':
    dof = upper_dof
else:
    dof = lower_dof


start_sample = 0
# q_cosmik= read_joint_angles_wholebody(path_cosmik, start_sample)
# q_mocap = read_joint_angles_wholebody(path_mocap, start_sample)


q_cosmik= read_specific_joint(path_cosmik,dof, start_sample)[10:]
q_mocap = read_specific_joint(path_mocap,dof, start_sample)[10:]

rmse_list = []
# Plot one figure per joint
# for i, name in enumerate(lower_dof):
    
#     rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
#     rmse = rmse * (180 / np.pi)
#     # print(rmse)
#     print(name , ':', rmse)
#     rmse_list.append(rmse)
#     plt.figure()
#     plt.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
#     plt.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
#     plt.title(f"{name} (RMSE: {rmse:.4f})")
#     plt.xlabel("samples")
#     plt.ylabel("Angle (rad)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


n_dofs = len(dof)
print(n_dofs)
first_batch = 6
remaining = n_dofs - first_batch
# First batch: plot first 6 as subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()  # make it easier to index

for i in range(first_batch):
    name = dof[i]
    rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
    rmse = rmse * (180 / np.pi)
    print(name, ':', rmse)
    rmse_list.append(rmse)
    
    ax = axes[i]
    ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
    ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
    ax.set_title(f"{name} (RMSE: {rmse:.4f})")
    ax.set_xlabel("samples")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

if remaining > 0:
    n_cols = 3
    n_rows = int(np.ceil(remaining / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for j in range(remaining):
        i = first_batch + j
        name = dof[i]
        rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
        rmse = rmse * (180 / np.pi)
        print(name, ':', rmse)
        rmse_list.append(rmse)
        
        ax = axes[j]
        ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
        ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
        ax.set_title(f"{name} (RMSE: {rmse:.4f})")
        ax.set_xlabel("samples")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True)
        ax.legend()

    # Hide unused subplots if any
    for j in range(remaining, len(axes)):
        fig.delaxes(axes[j])

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