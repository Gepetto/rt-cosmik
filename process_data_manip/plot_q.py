#plot q_cosmik and q_mocap to check if i have same pattern
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint

no_trial = "Nicolas"
task = "robot_polissage"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_mocap_ipopt.csv"

path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_cosmik_ipopt.csv" 

dofs  =  ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']

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

dof_to_plot = 'all'
if dof_to_plot =='upper':
    print('ok')
    dof = upper_dof
elif dof_to_plot == 'lower':
    dof = lower_dof
else:
    dof=dofs


start_sample = 0
# q_cosmik= read_joint_angles_wholebody(path_cosmik, start_sample)
# q_mocap = read_joint_angles_wholebody(path_mocap, start_sample)


q_cosmik= read_specific_joint(path_cosmik,dof, start_sample)[10:]
q_mocap = read_specific_joint(path_mocap,dof, start_sample)[10:]

# rmse_list = []
# Plot one figure per joint
# for i, name in enumerate(dof):
    
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


rmse_list = []
n_per_fig = 6  # Number of subplots per figure

for j, i in enumerate(range(7, len(dof))):
    name = dof[i]
    rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
    rmse = rmse * (180 / np.pi)
    rmse_list.append(rmse)

    # Create a new figure every 6 plots
    if j % n_per_fig == 0:
        fig, axs = plt.subplots(n_per_fig, 1, figsize=(8, 12))
        fig.tight_layout(pad=4.0)
    
    ax = axs[j % n_per_fig]
    ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
    ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
    ax.set_title(f"{name} (RMSE: {rmse:.4f})")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

    # Show the figure after every 6 plots or at the end
    if (j % n_per_fig == n_per_fig - 1) or (i == len(dof) - 1):
        plt.show()


# n_dofs = len(dof)
# print(n_dofs)
# first_batch = 6
# remaining = n_dofs - first_batch
# # First batch: plot first 6 as subplots
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()  # make it easier to index

# for i in range(first_batch):
#     name = dof[i]
#     rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
#     rmse = rmse * (180 / np.pi)
#     print(name, ':', rmse)
#     rmse_list.append(rmse)
    
#     ax = axes[i]
#     ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
#     ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
#     ax.set_title(f"{name} (RMSE: {rmse:.4f})")
#     ax.set_xlabel("samples")
#     ax.set_ylabel("Angle (rad)")
#     ax.grid(True)
#     ax.legend()

# plt.tight_layout()
# plt.show()

# if remaining > 0:
#     n_cols = 3
#     n_rows = int(np.ceil(remaining / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
#     axes = axes.flatten()

#     for j in range(remaining):
#         i = first_batch + j
#         name = dof[i]
#         rmse = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
#         rmse = rmse * (180 / np.pi)
#         print(name, ':', rmse)
#         rmse_list.append(rmse)
        
#         ax = axes[j]
#         ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='blue')
#         ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
#         ax.set_title(f"{name} (RMSE: {rmse:.4f})")
#         ax.set_xlabel("samples")
#         ax.set_ylabel("Angle (rad)")
#         ax.grid(True)
#         ax.legend()

#     # Hide unused subplots if any
#     for j in range(remaining, len(axes)):
#         fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()
average_rmse = np.mean(rmse_list)
print(f"\nAverage RMSE across all joints: {average_rmse:.4f} deg")
