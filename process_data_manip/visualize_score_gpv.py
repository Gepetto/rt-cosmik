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



base_path = "/home/ngouget/Codes"
subject = "Kahina"
trial = "overhead"
trial_path = os.path.join(base_path, f"datasets/COSMIK_dataset_mixed/{subject}/{trial}")


# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mocap_downsampled_to_40hz.csv"
# df_wide = pd.read_csv(path_to_csv)
# df_wide.columns = [col.replace(f"{no_trial}:", "") for col in df_wide.columns]
# frames = df_wide["Frame"] if "Frame" in df_wide.columns else range(len(df_wide))
# mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df_wide.columns if "_x" in col))



path_to_csv_mocap = os.path.join(trial_path, f"{trial}_mks_rt.csv")
path_to_kpt = os.path.join(trial_path, f"{trial}_jcp_hpe.csv")
path_to_score0 = os.path.join(trial_path, f"{trial}_camera_0.csv")
path_to_score2 = os.path.join(trial_path, f"{trial}_camera_2.csv")

score0 = pd.read_csv(path_to_score0)
score2 = pd.read_csv(path_to_score2)

score0 = score0.iloc[:,0]
score2 = score2.iloc[:,0]

two_scores = np.array([score0, score2])
print(two_scores)

min_score = np.min(two_scores, axis=0)

print(f"min score: {min_score}")


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

keypoints = pd.read_csv(path_to_kpt) 
keys_to_add = jcp_kpt

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

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
    viz.viewer.gui.addSphere(sphere_name, 0.01, [255, 0, 0, 1])
    viz.viewer.gui.setLightingMode(f'world/mocap_{name}', 'OFF')

for name in start_sample_mks.keys():
    sphere_name = f'world/mocap_{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 0, 0, 1.])
    viz.viewer.gui.setLightingMode(f'world/mocap_{name}', 'OFF')


for i in range(min(len(keypoints), len(result_markers))):
    for mks in markers_to_display:

        pos_mocap = result_markers[i][mks].reshape(3,)  # shape (3,)
        place(viz, f'world/mocap_{mks}', pin.SE3(np.eye(3), pos_mocap))

    for mks in jcp_kpt_to_display:
        pos_hpe_x = keypoints[f"{mks}_x"].values[i]/1.0
        pos_hpe_y = keypoints[f"{mks}_y"].values[i]/1.0
        pos_hpe_z = keypoints[f"{mks}_z"].values[i]/1.0
        place(viz, f'world/tri_{mks}', pin.SE3(np.eye(3), np.array([pos_hpe_x, pos_hpe_y, pos_hpe_z])))
        viz.viewer.gui.setColor(f'world/tri_{mks}', [255-(((float(min_score[i])*255-180))*255)/100, (((float(min_score[i])*255-180))*255)/100, 0, 1])
        print((((float(min_score[i])*255-180))*255)/100)
    
    time.sleep(0.03)