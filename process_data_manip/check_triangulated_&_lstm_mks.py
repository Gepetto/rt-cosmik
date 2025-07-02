import sys, os, pinocchio as pin, time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import pandas as pd
from viz_utils import place

# === Setup
no_trial = "Maxime"
task = "static"
csv_1_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/augmented_markers_4.csv"
csv_2_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/3d_keypoints_filtered_4.csv"

mks_names_1 = [
           'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

mks_names_2 = [
    "Nose", "LEye", "REye", "LEar", "REar", 
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]

def csv_to_dict_of_dicts(df, headers):
    result = {}
    for i, header in enumerate(headers):
        result[header] = {
            'x': df.iloc[:, i*3].tolist(),
            'y': df.iloc[:, i*3 + 1].tolist(),
            'z': df.iloc[:, i*3 + 2].tolist()
        }
    return result

def load_marker_dict(path, names):
    df = pd.read_csv(path, skiprows=1)
    assert len(df.columns) == 3 * len(names), f"Mismatch in columns for {path}"
    return csv_to_dict_of_dicts(df, names), len(df)

# === Load both marker sets
dict_1, n_frames_1 = load_marker_dict(csv_1_path, mks_names_1)
dict_2, n_frames_2 = load_marker_dict(csv_2_path, mks_names_2)
num_frames = min(n_frames_1, n_frames_2)

# === Init Viz
viz = GepettoVisualizer()
viz.initViewer()
viz.loadViewerModel("pinocchio")
viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.zeros((3, 1))))

# === Add all spheres
for name in mks_names_1:
    # color = [1, 0, 0, 1] if "wrist" in name or "knee" in name or "ankle" in name else [0, 0, 1, 1]
    sphere_name = f"world/lstm_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [1, 0, 0, 1])

for name in mks_names_2:
    sphere_name = f"world/tri_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 1, 0, 1])

# === Animation
for i in range(num_frames):
    for name in mks_names_1:
        pos = np.array([dict_1[name]['x'][i], dict_1[name]['y'][i], dict_1[name]['z'][i]])
        place(viz, f'world/lstm_{name}', pin.SE3(np.eye(3), pos.reshape(3, 1)))
    for name in mks_names_2:
        pos = np.array([dict_2[name]['x'][i], dict_2[name]['y'][i], dict_2[name]['z'][i]])
        place(viz, f'world/tri_{name}', pin.SE3(np.eye(3), pos.reshape(3, 1)))
    time.sleep(0.03)
