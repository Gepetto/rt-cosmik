import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from viz_utils import place
import pandas as pd 
from src.rtcosmik.utils.read_write_utils import parse_marker_csv,udp_csv_to_dataframe,read_mks_data,load_transformation,plot_marker_comparison
from collections import defaultdict

p = argparse.ArgumentParser(description="calculate rmse over all markers and frames with augmented local data")
p.add_argument('--use-mocap', choices=['T','F'], default='T', required=True)        # must be 'T' for this script (mocap JCP + mocap GT)
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--fine-tune', choices=['T','F'], default='F')
p.add_argument('--add-layer', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--rot-prob', type=float, default=0.0, help='Probability to apply a random yaw rotation per window (0..1)')
p.add_argument('--rot-max-deg', type=float, default=30.0, help='Max absolute rotation in degrees (uniform in [-max, max])')
p.add_argument('--up-axis', choices=['y','z'], default='z', help='Which axis is vertical in your data (usually y or z)')
p.add_argument('--rotation-scheme', choices=['off','prob','det'], default='off',
               help="off: no rotation; prob: single random yaw per window using --rot-prob/--rot-max-deg; det: emit n evenly-spaced yaws per window")
p.add_argument('--n-rotations', type=int, default=1,
               help="When --rotation-scheme det, emit this many evenly-spaced yaw angles per window (full circle).")
p.add_argument('--subject', type=str, default=None)
p.add_argument('--trial', type=str, default=None)

args = p.parse_args()

base_path = "/home/ngouget/Codes"

subject_path = f"/home/ngouget/Codes/datasets/COSMIK_dataset_mixed/{args.subject}"
trial_path = os.path.join(subject_path, args.trial)

# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mocap_downsampled_to_40hz.csv"
# df_wide = pd.read_csv(path_to_csv)
# df_wide.columns = [col.replace(f"{no_trial}:", "") for col in df_wide.columns]
# frames = df_wide["Frame"] if "Frame" in df_wide.columns else range(len(df_wide))
# mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df_wide.columns if "_x" in col))

if args.use_mocap == "T":
    path_to_csv_mocap = os.path.join(trial_path, f"{args.trial}_trajectories.csv")
    path_to_csv_lstm = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.csv")
    path_to_csv_lstm_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_OpenCap.csv")
    path_to_kpt = os.path.join(trial_path, f"{args.trial}_jcp_mocap.csv")
elif args.use_mocap == "F":
    path_to_csv_mocap = os.path.join(trial_path, f"{args.trial}_mks_rt.csv")
    path_to_csv_lstm = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_hpe_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.csv")
    path_to_csv_lstm_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_hpe_OpenCap.csv")
    path_to_kpt = os.path.join(trial_path, f"3D_keypoints_fused_begin.csv")
else:
    raise Exception("Use mocap not supported. Please select T or F.")

# path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/joint_center_positions.csv"

marker_mocap_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis'] #mocap data
mks_names = marker_mocap_names
# df_wide = marker_data_to_dataframe(df_raw,mks_names)
if args.use_mocap == "T":
    df_wide = pd.read_csv(path_to_csv_mocap)
    result_markers, start_sample_mks = read_mks_data(df_wide, converter = 1000.0)
elif args.use_mocap == "F":
    df_wide = pd.read_csv(path_to_csv_mocap)
    result_markers, start_sample_mks = read_mks_data(df_wide, converter = 1.0)

jcp_kpt = [
    "Nose", "LEye", "REye", "LEar", "REar", 
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]
jcp_kpt_to_display = [
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Neck",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]
lstm_mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

markers_to_display = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study',
           'r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']
# lstm_mks_names = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","Rknee","LAnkle","RAnkle","Head","Neck","Hip","LBigToe","RBigToe","LSmallToe", "RSmallToe", "LHeel","RHeel"]



data_markers_lstm = pd.read_csv(path_to_csv_lstm)
data_markers_lstm_OpenCap = pd.read_csv(path_to_csv_lstm_OpenCap)
keypoints = pd.read_csv(path_to_kpt) 
keys_to_add = jcp_kpt

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

result_markers_lstm, start_sample_lstm = read_mks_data(data_markers_lstm, converter = 1.0)
result_markers_lstmOpenCap, start_sample_lstm_OpenCap = read_mks_data(data_markers_lstm_OpenCap, converter = 1.0)

# plot_marker_comparison(result_markers, result_markers_lstm, markers_to_plot=markers_to_display)
# === Initialiser le visualiseur Gepetto ===
viz = GepettoVisualizer()
try:
    viz.initViewer()
except ImportError as err:
    print("Install gepetto-viewer.")
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Start gepetto-viewer before running this script.")
    sys.exit(0)

viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.zeros((3,1))))

for name in jcp_kpt_to_display:
    sphere_name = f"world/tri_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0, 255, 1])

for name in start_sample_lstm_OpenCap.keys():
    sphere_n = f'world/lstm_{name}'
    viz.viewer.gui.addSphere(sphere_n, 0.015, [0, 0, 0, 1.])

for name in start_sample_lstm.keys():
    sphere_n = f'world/lstm_nominal_{name}'
    viz.viewer.gui.addSphere(sphere_n, 0.015, [0, 255, 0, 1.])

for name in start_sample_mks.keys():
    sphere_name = f'world/mocap_{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [255, 0, 0, 1.])

squared_errors = defaultdict(list)
all_squared_errors = []  

squared_errors_OpenCap = defaultdict(list)
all_squared_errors_OpenCap = []

for i in range(min(len(result_markers), len(result_markers_lstm))):
    for mks in markers_to_display:
        pos_mocap = result_markers[i][mks].reshape(3,)  # shape (3,)
        pos_mks = result_markers_lstm[i][mks].reshape(3,)  # shape (3,)
        pos_mks_OpenCap = result_markers_lstmOpenCap[i][mks].reshape(3,)  # shape (3,)

        place(viz, f'world/mocap_{mks}', pin.SE3(np.eye(3), pos_mocap))
        place(viz, f'world/lstm_nominal_{mks}', pin.SE3(np.eye(3), pos_mks))
        place(viz, f'world/lstm_{mks}', pin.SE3(np.eye(3), pos_mks_OpenCap))

        error = np.linalg.norm(pos_mocap - pos_mks)  # Euclidean distance
        squared_errors[mks].append(error)
        all_squared_errors.append(error)

        error_OpenCap = np.linalg.norm(pos_mocap - pos_mks_OpenCap)  # Euclidean distance
        squared_errors_OpenCap[mks].append(error_OpenCap)
        all_squared_errors_OpenCap.append(error_OpenCap)

    if args.use_mocap == "T":
        for mks in jcp_kpt_to_display:
            pos_hpe_x = keypoints[f"{mks}_x"].values[i]/1000.0
            pos_hpe_y = keypoints[f"{mks}_y"].values[i]/1000.0
            pos_hpe_z = keypoints[f"{mks}_z"].values[i]/1000.0
            place(viz, f'world/tri_{mks}', pin.SE3(np.eye(3), np.array([pos_hpe_x, pos_hpe_y, pos_hpe_z])))
    elif args.use_mocap == "F":
        for mks in jcp_kpt_to_display:
            pos_hpe_x = keypoints[f"{mks}_x"].values[i]/1.0
            pos_hpe_y = keypoints[f"{mks}_y"].values[i]/1.0
            pos_hpe_z = keypoints[f"{mks}_z"].values[i]/1.0
            place(viz, f'world/tri_{mks}', pin.SE3(np.eye(3), np.array([pos_hpe_x, pos_hpe_y, pos_hpe_z])))
    else:
        raise Exception("Input type not supported. Please select T or F.")

    
    time.sleep(0.03)

# Compute RMSE per marker
rmse_per_marker = {}
for mks, errors in squared_errors.items():
    mse = np.mean(errors)
    rmse = mse
    rmse_per_marker[mks] = rmse

rmse_per_marker_OpenCap = {}
for mks, errors in squared_errors_OpenCap.items():
    mse = np.mean(errors)
    rmse = mse
    rmse_per_marker_OpenCap[mks] = rmse

# Print or log RMSE
for mks, rmse in rmse_per_marker.items():
    print(f"RMSE for marker {mks}: {rmse:.3f} m")

average_rmse = np.mean(all_squared_errors)
print(f"\nAverage RMSE over all markers and frames: {average_rmse:} m")

print("rmse lstm finetuned ##############################################################")
# Print or log RMSE
for mks, rmse in rmse_per_marker_OpenCap.items():
    print(f"OpenCap RMSE for marker {mks}: {rmse:.3f} m")

average_rmse_OpenCap = np.mean(all_squared_errors_OpenCap)
print(f"\nOpenCap Average RMSE over all markers and frames: {average_rmse_OpenCap:} m")

# 🔑 ligne spéciale pour parsing
print(f"RESULTS,{average_rmse:.6f},{average_rmse_OpenCap:.6f}")
