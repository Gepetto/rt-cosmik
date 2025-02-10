import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# y = pd.read_csv('q_cosmik_qp.csv') #wrist et ipopt pareil (mocap)
y2 = pd.read_csv('q_cosmik_qp_modele_mocap.csv')
dofs = [
    'FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3',
    "middle_lumbar_Z", "middle_lumbar_Y", "right_shoulder_Z", "right_shoulder_X",
    "right_shoulder_Y", "right_elbow_Z", "right_elbow_Y", "left_shoulder_Z",
    "left_shoulder_X", "left_shoulder_Y", "left_elbow_Z", "left_elbow_Y",
    "right_hip_Z", "right_hip_X", "right_hip_Y", "right_knee_Z", "right_ankle_Z",
    "left_hip_Z", "left_hip_X", "left_hip_Y", "left_knee_Z", "left_ankle_Z"
]
print(y2.shape[1])
for i in range(0,y2.shape[1]):
    print(i)
    plt.figure()
    # plt.title(f"{'_'.join(y.columns[i])}")
    # plt.plot(y.values[:, i], 'orange', label='mocap_qp', linewidth=3)
    plt.plot(y2.values[:,i], 'green', label='cosmik_qp.csv')  # Adjust index by subtracting 2
    plt.ylabel(f'{dofs[i]}')
    plt.legend()
    plt.grid()
    plt.show()



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