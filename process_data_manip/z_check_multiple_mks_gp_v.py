#allows to check mks mocap vs mks lstm and jcp mks
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
from src.rtcosmik.utils.read_write_utils import parse_marker_csv,udp_csv_to_dataframe,read_mks_data,load_transformation,plot_marker_comparison, read_mocap_data
from collections import defaultdict

idx = "4279"
no_trial = "Alessandro"
no_trial_c = "alessandro"
task = "bolting"
path_to_csv_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/mocap/mocap_{no_trial}/{task}/mocap_downsampled_to_40hz.csv"
df_wide = pd.read_csv(path_to_csv_mocap).iloc[:, 1:]
result_markers, start_sample_mks = read_mks_data(df_wide, converter = 1000.0)


path_to_csv_lstm = f"/root/workspace/ros_ws/src/rt-cosmik/output/cosmik/cosmik_{no_trial_c}/{task}/augmented_markers.csv"
data_markers_lstm = pd.read_csv(path_to_csv_lstm)
result_markers_lstm, start_sample_lstm = read_mks_data(data_markers_lstm, converter = 1.0)


path_to_csv_lstm2 = f"/root/workspace/ros_ws/src/rt-cosmik/output/cosmik_jcp/{no_trial}/{task}/augmented_markers_jcp_corrected_finetuned.csv"
data_markers_lstm2 = pd.read_csv(path_to_csv_lstm2)
result_markers_lstm2, start_sample_lstm2 = read_mks_data(data_markers_lstm2, converter = 1.0)


path_to_kpt_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/cosmik_jcp/{no_trial}/{task}/{task}_jcp_hpe_filtered_2_corrected.csv"
data_markers_hpe_kpt_mocap = pd.read_csv(path_to_kpt_mocap) 
result_markers_hpe_kpt, start_sample_hpe_kpt = read_mks_data(data_markers_hpe_kpt_mocap, converter = 1.0)


marker_mocap_names = ['RPSI', 'LPSI', 'RASI', 'LASI', 'RPELV', 'LPELV', 'RKNE', 'RMKNE', 'RTHI1', 'RANK', 'RMANK', 'RTIB1', 'RHEE', 'R5MHD', 'RTOE',
                       'LKNE', 'LMKNE', 'LTHI1', 'LANK', 'LMANK', 'LTIB1', 'LHEE', 'L5MHD', 'LTOE', 'RSHO', 'LSHO', 'SJN', 'C7', 'TV8', 'STRN', 
                       'TV12', 'RELB', 'RMELB', 'RUARM', 'RMWRI', 
                      'RWRI', 'RFORE', 'RHM2', 'RHM5', 'RHAND', 'LELB', 'LMELB', 'LUARM', 'LMWRI', 
                      'LWRI', 'LFORE', 'LHM2', 'LHM5', 'LHAND', 'RHD', 'LHD', 'FHD', 'BHD'] #mocap data

mocap_to_display_map = {
    'RASI': 'r.ASIS_study',
    'LASI': 'L.ASIS_study',
    'RPSI': 'r.PSIS_study',
    'LPSI': 'L.PSIS_study',

    'RKNE': 'r_knee_study',
    'RMKNE': 'r_mknee_study',
    'RANK': 'r_ankle_study',
    'RMANK': 'r_mankle_study',
    'RTOE': 'r_toe_study',
    'R5MHD': 'r_5meta_study',
    'RHEE': 'r_calc_study',

    'LKNE': 'L_knee_study',
    'LMKNE': 'L_mknee_study',
    'LANK': 'L_ankle_study',
    'LMANK': 'L_mankle_study',
    'LTOE': 'L_toe_study',
    'L5MHD': 'L_5meta_study',
    'LHEE': 'L_calc_study',

    'RSHO':'r_shoulder_study',
    'LSHO':'L_shoulder_study',
    'C7' : 'C7_study',

    'RELB':'r_lelbow_study',
    'RMELB': 'r_melbow_study',

    'LELB':'L_lelbow_study',
    'LMELB': 'L_melbow_study',

    'RWRI':'r_lwrist_study',
    'RMWRI':'r_mwrist_study',

    'LWRI':'L_lwrist_study',
    'LMWRI':'L_mwrist_study',
}
print("ok")
result_markers_mapped = []
for frame_dict in result_markers:
    new_frame_dict = {}
    for old_name, value in frame_dict.items():
        new_name = mocap_to_display_map.get(old_name, old_name) 
        new_frame_dict[new_name] = value
    result_markers_mapped.append(new_frame_dict)
print("mapped")

mks_names = marker_mocap_names
hpe_kpt = [
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle",
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



# columns_to_sadd = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

# data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)


rmse_results, rmse_mean_per_dataset = plot_marker_comparison(
    datasets=[result_markers_mapped, result_markers_lstm,result_markers_lstm2],
    labels=["mocap", "opencap", "finetuned"],
    markers_to_plot=markers_to_display,
    ref_idx=0,
    colors=["red", "green","blue"],
    show_barplot=True  
)
print(rmse_mean_per_dataset)
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

for name in start_sample_hpe_kpt.keys():
    sphere_name = f"world/tri_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [255, 255, 0, 1])

# for name in start_sample_hpe_kpt_mocap.keys():
#     sphere_name = f"world/tri_{name}_offset"
#     viz.viewer.gui.addSphere(sphere_name, 0.01, [255, 255, 255, 1])

for name in start_sample_lstm2.keys():
    sphere_n = f'world/lstm_finetuned_{name}'
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
for i in range(len(result_markers_mapped)):
    for mks in markers_to_display:
        pos_mocap = result_markers_mapped[i][mks].reshape(3,)  # shape (3,)
        pos_mks = result_markers_lstm[i][mks].reshape(3,)  # shape (3,)
        pos_mks2 = result_markers_lstm2[i][mks].reshape(3,)  # shape (3,)

        place(viz, f'world/mocap_{mks}', pin.SE3(np.eye(3), pos_mocap))
        place(viz, f'world/lstm_nominal_{mks}', pin.SE3(np.eye(3), pos_mks))
        place(viz, f'world/lstm_finetuned_{mks}', pin.SE3(np.eye(3), pos_mks2))

        error = np.linalg.norm(pos_mocap - pos_mks)  # Euclidean distance
        squared_errors[mks].append(error**2)
        all_squared_errors.append(error**2)

        error2 = np.linalg.norm(pos_mocap - pos_mks2)  # Euclidean distance
        squared_errors2[mks].append(error2**2)
        all_squared_errors2.append(error2**2)

    
    for mks in start_sample_hpe_kpt.keys():
        pos_hpe = result_markers_hpe_kpt[i][mks].reshape(3,)
        place(viz, f'world/tri_{mks}', pin.SE3(np.eye(3), pos_hpe))

        # pos_hpe_mocap = result_markers_hpe_kpt_mocap[i][mks].reshape(3,)
        # place(viz, f'world/tri_{mks}_offset', pin.SE3(np.eye(3), pos_hpe_mocap))

    
    time.sleep(0.05)
    # input()

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