#plot q_cosmik and q_mocap to check if i have same pattern
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlation_lags
from scipy.signal import correlate


no_trial = "4279"
task = "robot_welding"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/q_mocap.csv"
path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/q_cosmik_fused_all.csv"

dofs  =  ['Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                        #   'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi', #'Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', #'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']

upper_dof = ['Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
                          'Lshoulder_flex_ext',
                          'Lshoulder_abd_add', 'Lshoulder_int_ext_rot',
                          'Relbow_flex_ext', 'Relbow_pron_supi', 'Lelbow_flex_ext',
                          'Lelbow_pron_supi']
                          
lower_dof=['Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Lknee_flex_ext', 'Lankle_flex_ext', 'Rankle_abd_add','Lankle_abd_add']

#opencap : hips(23),knees(23), ankle(2*2) et lumbar(3) + freeflyer
dof_opencap = [ 'Lumbar_flex_ext', 'Lumbar_lateral_flex', 
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext','Rankle_abd_add', 'Lknee_flex_ext', 'Lankle_flex_ext','Lankle_abd_add']
dof_to_plot = 'all'
start_dof = 0

if dof_to_plot =='upper':
    dof = upper_dof
elif dof_to_plot == 'lower':
    dof = lower_dof
elif dof_to_plot =='opencap':
    dof = dof_opencap
else:
    dof=dofs


def synchronize_signals(sig1, sig2):
    """
    Synchronize two signals by shifting sig2 relative to sig1.

    Args:
        sig1: numpy array, reference signal
        sig2: numpy array, signal to be shifted

    Returns:
        lag: number of samples sig2 was shifted (+ means sig2 delayed)
    """

    corr = correlate(sig1, sig2, mode="full")
    lags = correlation_lags(len(sig1), len(sig2), mode="full")
    lag = lags[np.argmax(corr)]
    return lag

df_cosmik = pd.read_csv(path_cosmik).iloc[:, 7:]
df_mocap  = pd.read_csv(path_mocap).iloc[:, 7:]
# df_mocap = df_mocap.iloc[280::2]

if df_cosmik.shape[0] > df_mocap.shape[0]:
            df_cosmik = df_cosmik.iloc[:-1, :]
elif df_cosmik.shape[0] < df_mocap.shape[0]:
    df_mocap = df_mocap.iloc[:-1, :]

# Use knee angle to compute lag
knee_cosmik = df_cosmik["Rknee_flex_ext"].values
knee_mocap  = df_mocap["Rknee_flex_ext"].values
lag = synchronize_signals(knee_cosmik, knee_mocap)
print("lag",lag)

start_sample = 0
quat = ['FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw']
quaternion_cosmik = read_specific_joint(path_cosmik,quat, start_sample)
r_cosmik = R.from_quat(quaternion_cosmik)
euler_angles_rad_cosmik = r_cosmik.as_euler('xyz', degrees=False)

quaternion_mocap = read_specific_joint(path_mocap,quat, start_sample)
r_mocap = R.from_quat(quaternion_mocap)
euler_angles_rad_mocap = r_mocap.as_euler('xyz', degrees=False)

q_cosmik= read_specific_joint(path_cosmik,dof, start_sample)
q_mocap = read_specific_joint(path_mocap,dof, start_sample)
# q_mocap = q_mocap[280::2]
# q_cosmik = q_cosmik[280::2]
rmse_list = []
corr_list = []
mae_list =  []

excluded_joints = ['Lwrist_flex_ext', 'Lwrist_x', 'Rwrist_flex_ext', 'Rwrist_x','Lelbow_pron_supi','Relbow_pron_supi']

# Filter the indices of joints to include
joint_indices = [i for i in range(start_dof, len(dof)) if dof[i] not in excluded_joints]
n_per_fig = 6  # Number of subplots per figure

if lag > 0:
    q_cosmik = q_cosmik[lag:]
    q_mocap = q_mocap[:len(q_cosmik)]  # truncate Cosmik accordingly


for j, i in enumerate(joint_indices):
    name = dof[i]
    
    rmse_rad = np.sqrt(np.mean((q_mocap[:, i] - q_cosmik[:, i]) ** 2))
    rmse = rmse_rad * (180 / np.pi)
    rmse_list.append(rmse)

    # MAE
    mae_rad = np.mean(np.abs(q_mocap[:, i] - q_cosmik[:, i]))
    mae = mae_rad * (180 / np.pi)
    mae_list.append(mae)


    # Compute Pearson correlation coefficient
    corr_coef = np.corrcoef(q_mocap[:, i], q_cosmik[:, i])[0, 1]
    corr_list.append(corr_coef)

    # Create a new figure every 6 plots
    if j % n_per_fig == 0:
        fig, axs = plt.subplots(n_per_fig, 1, figsize=(8, 12))
        fig.tight_layout(pad=4.0)
    
    ax = axs[j % n_per_fig]
    ax.plot(q_cosmik[:, i], label="Cosmik", linewidth=2, color='green')
    ax.plot(q_mocap[:, i], label="Mocap", linewidth=2, color='red')
    ax.set_title(f"{name} RMSE: {rmse:.4f}deg, {rmse_rad:.4f}rad, MAE: {mae:.2f}° ,Corr: {corr_coef:.2f})")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

    # Show the figure after every 6 plots or at the end
    if (j % n_per_fig == n_per_fig - 1) or (j == len(joint_indices) - 1):
        plt.show()

joint_names = dof[start_dof:]
joint_names = [name for name in joint_names if name not in excluded_joints]
rmse_array = np.array(rmse_list)
avg_rmse = np.mean(rmse_array)

# Bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(joint_names, rmse_array, color='skyblue', edgecolor='black')

# Add average line
plt.axhline(avg_rmse, color='red', linestyle='--', label=f'Average RMSE: {avg_rmse:.2f}°')

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}", 
             ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.ylabel("RMSE (degrees)")
plt.title("Joint Angle RMSEs")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

rmse_array = np.array(rmse_list)
std_rmse = np.std(rmse_array)

average_rmse = np.mean(rmse_list)
avg_corr = np.mean(corr_list)
avg_mae = np.mean(mae_list)
print(f"\nAverage RMSE across all joints: {average_rmse:.4f} deg, {std_rmse:.4f}")
print(f"\nAverage mae across all joints: {avg_mae:.4f} deg")

print(f"\nAverage cc across all joints: {avg_corr:.4f} ")
