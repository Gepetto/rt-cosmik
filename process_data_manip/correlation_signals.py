import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate


dofs = [
    'FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3',
    "middle_lumbar_Z", "middle_lumbar_Y", "right_shoulder_Z", "right_shoulder_X",
    "right_shoulder_Y", "right_elbow_Z", "right_elbow_Y", "left_shoulder_Z",
    "left_shoulder_X", "left_shoulder_Y", "left_elbow_Z", "left_elbow_Y",
    "right_hip_Z", "right_hip_X", "right_hip_Y", "right_knee_Z", "right_ankle_Z",
    "left_hip_Z", "left_hip_X", "left_hip_Y", "left_knee_Z", "left_ankle_Z"
]
mocap = pd.read_csv('q/q_mocap_qp_downsampled_33Hz.csv')
lstm = pd.read_csv('q/q_cosmik_qp_interpolated_33Hz.csv')


mks_mocap = mocap.iloc[:, 2].values  #2,3,4,5,6,7,8
mks_lstm = lstm.iloc[:, 1].values  #1,2,3,4,5,6,7

# Compute cross-correlation
correlation = correlate(mks_mocap, mks_lstm, mode='full')
lags = np.arange(-len(mks_lstm) + 1, len(mks_mocap))

# Find the lag with the maximum correlation
max_corr_index = np.argmax(correlation)
sample_difference = lags[max_corr_index]

print(f"Maximum correlation occurs at a lag of {sample_difference} samples.")

# Plot the signals
plt.figure(figsize=(10, 5))

# Plot Signal 1
plt.plot(mks_mocap[32:], label='mocap', linestyle='-')

# Plot Signal 2
plt.plot(mks_lstm, label='lstm', linestyle='--')

# Add labels and legend
plt.title('q')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
plt.legend()
plt.grid()

# Show the plot
plt.show()


rmse_values = []

for i in range(2, mocap.shape[1]):
    # Calculate RMSE for each DOF
    mocap_values = mocap.values[ :, i]  # Get mocap values
    lstm_values = lstm.values[:, i-1]    # Get LSTM values
    
    # Calculate RMSE for this DOF
    # rmse = ((np.sqrt(np.mean((mocap_values - lstm_values) ** 2))) * 180 ) / np.pi
    # rmse_values.append(rmse)
    
    # Plot for this DOF
    plt.figure()
    plt.plot(mocap_values, label='q_mocap', linestyle='-')
    plt.plot(lstm_values, label='q_lstm', linestyle='--')
    plt.ylabel(f'{dofs[i - 2]} (rad)')
    # plt.title(f"RMSE = {rmse:.4f} (deg)")
    plt.legend()
    plt.grid()
    plt. show()

# Calculate and print the average RMSE
average_rmse = np.mean(rmse_values)
print(f'Average RMSE across all DOFs: {average_rmse}')