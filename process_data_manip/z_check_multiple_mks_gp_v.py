import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from viz_utils import place
import pandas as pd 
from src.rtcosmik.utils.read_write_utils import parse_marker_csv,udp_csv_to_dataframe,read_mks_data,load_transformation,plot_marker_comparison
from collections import defaultdict

nbr_cam = 2
no_trial = "Claire_"
task = "sanding"

path_to_csv_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mks_data_gapfilled.csv"
# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mocap_downsampled_to_40hz.csv"
# df_wide = pd.read_csv(path_to_csv)
# df_wide.columns = [col.replace(f"{no_trial}:", "") for col in df_wide.columns]
# frames = df_wide["Frame"] if "Frame" in df_wide.columns else range(len(df_wide))
# mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df_wide.columns if "_x" in col))

path_to_csv_lstm = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/augmented_markers_test.csv"
path_to_csv_lstm2 = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/augmented_markers_{nbr_cam}.csv"
path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/3d_keypoints_filtered_{nbr_cam}.csv"
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
df_wide = udp_csv_to_dataframe(path_to_csv_mocap, mks_names)
result_markers, start_sample_mks = read_mks_data(df_wide, converter = 1.0)

hpe_kpt = [
    "Nose", "LEye", "REye", "LEar", "REar", 
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]
lstm_mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
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
data_markers_lstm2 = pd.read_csv(path_to_csv_lstm2)
keypoints = pd.read_csv(path_to_kpt) 
keys_to_add = hpe_kpt

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

result_markers_lstm, start_sample_lstm = read_mks_data(data_markers_lstm, converter = 1.0)
result_markers_lstm2, start_sample_lstm2 = read_mks_data(data_markers_lstm2, converter = 1.0)

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

for name in hpe_kpt:
    sphere_name = f"world/tri_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0, 255, 1])

for name in start_sample_lstm2.keys():
    sphere_n = f'world/lstm_{name}'
    viz.viewer.gui.addSphere(sphere_n, 0.015, [0, 0, 255, 1.])

for name in start_sample_lstm.keys():
    sphere_n = f'world/lstm_nominal_{name}'
    viz.viewer.gui.addSphere(sphere_n, 0.015, [0, 255, 0, 1.])

for name in start_sample_mks.keys():
    sphere_name = f'world/mocap_{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [255, 0, 0, 1.])

squared_errors = defaultdict(list)
all_squared_errors = []  

squared_errors2 = defaultdict(list)
all_squared_errors2 = [] 

for i in range(len(result_markers)):
    for mks in markers_to_display:
        pos_mocap = result_markers[i][mks].reshape(3,)  # shape (3,)
        pos_mks = result_markers_lstm[i][mks].reshape(3,)  # shape (3,)
        pos_mks2 = result_markers_lstm2[i][mks].reshape(3,)  # shape (3,)

        place(viz, f'world/mocap_{mks}', pin.SE3(np.eye(3), pos_mocap))
        place(viz, f'world/lstm_nominal_{mks}', pin.SE3(np.eye(3), pos_mks))
        place(viz, f'world/lstm_{mks}', pin.SE3(np.eye(3), pos_mks2))

        error = np.linalg.norm(pos_mocap - pos_mks)  # Euclidean distance
        squared_errors[mks].append(error**2)
        all_squared_errors.append(error**2)

        error2 = np.linalg.norm(pos_mocap - pos_mks2)  # Euclidean distance
        squared_errors2[mks].append(error2**2)
        all_squared_errors2.append(error2**2)

    
    # for mks in hpe_kpt:
    #     pos_hpe = result_markers_lstm[i][mks].reshape(3,)
    #     place(viz, f'world/tri_{mks}', pin.SE3(np.eye(3), pos_hpe))

    
    time.sleep(0.03)

# Compute RMSE per marker
rmse_per_marker = {}
for mks, errors in squared_errors.items():
    mse = np.mean(errors)
    rmse = np.sqrt(mse)
    rmse_per_marker[mks] = rmse

rmse_per_marker2 = {}
for mks, errors in squared_errors2.items():
    mse = np.mean(errors)
    rmse = np.sqrt(mse)
    rmse_per_marker2[mks] = rmse

# Print or log RMSE
for mks, rmse in rmse_per_marker.items():
    print(f"RMSE for marker {mks}: {rmse:.3f} m")

average_rmse = np.sqrt(np.mean(all_squared_errors))
print(f"\nAverage RMSE over all markers and frames: {average_rmse:} m")

print("rmse lstm finetuned ##############################################################")
# Print or log RMSE
for mks, rmse in rmse_per_marker2.items():
    print(f"RMSE for marker {mks}: {rmse:.3f} m")

average_rmse2 = np.sqrt(np.mean(all_squared_errors2))
print(f"\nAverage RMSE over all markers and frames: {average_rmse2:} m")