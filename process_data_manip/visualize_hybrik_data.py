import os
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data, read_subject_info
import pandas as pd
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker
from src.rtcosmik.config_loader import settings
from src.rtcosmik.ik.ik import RT_SWIKA
from collections import deque
from src.rtcosmik.human_model.urdf_model import *
import multiprocessing as mp

# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/'
rt_cosmik_path = os.path.dirname(script_directory)

no_trial = 'Nicolas'
task = 'bolting'

# markers to skip
mks_to_skip = ['TV8','TV12','SJN','STRN','LForearm','LUArm', 'RUArm','RHJC_study','LHJC_study',
               'LHand2','LHand1','LHL2','LHM5', 'RForearm','RHand2','RHand1','RHL2','RHM5']


print(f"\n=== Subject: {no_trial} | Task: {task} ===")

# read subject info
info_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/info.txt"
subject_height, subject_weight, gender = read_subject_info(info_path)

# input files
# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/hybrik/{task}/hybrik_camera_0_mks_lstm.csv"
path_to_csv_aligned = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/hybrik/{task}/hybrik_camera_0_mks_aligned.csv"
path_to_mocap_mks = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mks_model_mocap_downsampled.csv"

data_markers_lstm = pd.read_csv(path_to_csv_aligned)

start_sample=0
result_markers, start_sample_dict = read_mks_data(data_markers_lstm, start_sample=start_sample)

mks_names_1 = [
           'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study', 'Nose', 'Head', 'REar', 'LEar','REye','LEye']

mks_names_2 = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study','RHD','LHD','FHD','BHD', 'SJN','TV8','STRN','TV12'
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
# dict_1, n_frames_1 = load_marker_dict(path_to_csv, mks_names_1)
dict_2, n_frames_2 = load_marker_dict(path_to_mocap_mks, mks_names_2)
dict_3, n_frames_3 = load_marker_dict(path_to_csv_aligned, mks_names_1)

# load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True)
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

# scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender=gender,subject_height=subject_height)
print(human_model.nq)
human_model= mks_registration_hybrik(human_model,start_sample_dict, with_hand=False)
human_data = pin.Data(human_model)

################################################################# LOCK JOINTS 
all_joint_ids = set(range(1, human_model.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if human_model.existJointName(jn):
        joint_ids_to_lock.append(human_model.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(human_model)
# Build reduced model
human_model, human_visual_model = pin.buildReducedModel(
    human_model, human_visual_model, joint_ids_to_lock, q0)

print(human_model.nq)
human_data = pin.Data(human_model)

path_q = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/hybrik/{task}/q_hybrik_swika.csv"

q_hybrik = pd.read_csv(path_q,delimiter=',').to_numpy()

# VISUALISATION 
viz = GepettoVisualizer(human_model,human_collision_model,human_visual_model)

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

# === Add all spheres
# for name in mks_names_1:
#     # color = [1, 0, 0, 1] if "wrist" in name or "knee" in name or "ankle" in name else [0, 0, 1, 1]
#     sphere_name = f"world/hybrik_{name}"
#     viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0, 1, 1])

for name in mks_names_1:
    # color = [1, 0, 0, 1] if "wrist" in name or "knee" in name or "ankle" in name else [0, 0, 1, 1]
    sphere_name = f"world/hybrik_aligned_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 1, 0, 1])

for name in mks_names_2:
    # color = [1, 0, 0, 1] if "wrist" in name or "knee" in name or "ankle" in name else [0, 0, 1, 1]
    sphere_name = f"world/mocap_{name}"
    viz.viewer.gui.addSphere(sphere_name, 0.01, [1, 0, 0, 1])

for ii in range(len(dict_2['r.ASIS_study']['x'])  ):
    # for name in mks_names_1:
    #     pos = np.array([dict_1[name]['x'][ii], dict_1[name]['y'][ii], dict_1[name]['z'][ii]])
    #     place(viz, f'world/hybrik_{name}', pin.SE3(np.eye(3), pos.reshape(3, 1)))
    for name in mks_names_2:
        pos = np.array([dict_2[name]['x'][ii], dict_2[name]['y'][ii], dict_2[name]['z'][ii]])
        place(viz, f'world/mocap_{name}', pin.SE3(np.eye(3), pos.reshape(3, 1)))
    for name in mks_names_1:
        pos = np.array([dict_3[name]['x'][ii], dict_3[name]['y'][ii], dict_3[name]['z'][ii]])
        place(viz, f'world/hybrik_aligned_{name}', pin.SE3(np.eye(3), pos.reshape(3, 1)))
    q0 = pin.neutral(human_model)
    q0[:]=q_hybrik[ii,:]
    viz.display(q0)
    input()