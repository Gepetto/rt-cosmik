import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dofs = [
    'FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3',
    "middle_lumbar_Z", "middle_lumbar_Y", "right_shoulder_Z", "right_shoulder_X",
    "right_shoulder_Y", "right_elbow_Z", "right_elbow_Y", "left_shoulder_Z",
    "left_shoulder_X", "left_shoulder_Y", "left_elbow_Z", "left_elbow_Y",
    "right_hip_Z", "right_hip_X", "right_hip_Y", "right_knee_Z", "right_ankle_Z",
    "left_hip_Z", "left_hip_X", "left_hip_Y", "left_knee_Z", "left_ankle_Z"
]
mocap = pd.read_csv('q/q_mocap_qp_downsampled_33Hz.csv')
cosmik = pd.read_csv('q/q_cosmik_qp_interpolated_33Hz.csv')

rmse_values = []

for i in range(2, mocap.shape[1]):
    # Calculate RMSE for each DOF
    mocap_values = mocap.values[735 : mocap.shape[0] - 34, i]  # Get mocap values by removing the extra samples 735
    cosmik_values = cosmik.values[700:, i-1]    # Get cosmik values 700 pour virer partie non sychronisé
    
    # Calculate RMSE for this DOF
    rmse = ((np.sqrt(np.mean((mocap_values - cosmik_values) ** 2))) * 180 ) / np.pi
    rmse_values.append(rmse)
    
    # Plot for this DOF
    plt.figure()
    plt.plot(mocap_values, label='q_mocap', linestyle='-')
    plt.plot(cosmik_values, label='q_cosmik', linestyle='--')
    plt.ylabel(f'{dofs[i - 2]} (rad)')
    plt.title(f"RMSE = {rmse:.4f} (deg)")
    plt.legend()
    plt.grid()
    plt.show()

# Calculate and print the average RMSE
average_rmse = np.mean(rmse_values)
print(f'Average RMSE across all DOFs: {average_rmse}')

average_rmse = np.mean(rmse_values)
std_rmse = np.std(rmse_values)

# Plot bar graph for RMSE values
plt.figure(figsize=(12, 6))
plt.bar(dofs, rmse_values, color='skyblue')

# Add horizontal lines for average RMSE and standard deviation
plt.axhline(y=average_rmse, color='red', linestyle='--', linewidth=1.5, label=f'Average RMSE = {average_rmse:.2f} deg')
# plt.axhline(y=average_rmse + std_rmse, color='green', linestyle='--', linewidth=1, label=f'+1 Std = {average_rmse + std_rmse:.2f} deg')
# plt.axhline(y=average_rmse - std_rmse, color='green', linestyle='--', linewidth=1, label=f'-1 Std = {average_rmse - std_rmse:.2f} deg')

# Customize the plot
plt.xticks(rotation=90, fontsize=8)
plt.ylabel('RMSE (deg)')
plt.title('cosmik_model')
plt.legend(loc='upper right', fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()