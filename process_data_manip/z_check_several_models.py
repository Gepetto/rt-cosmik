import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/'
rt_cosmik_path = os.path.dirname(script_directory)
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data, udp_csv_to_dataframe,read_joint_angles_wholebody
import pandas as pd
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.pin_model import build_model
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from src.rtcosmik.ik.ik import RT_IK,RT_SWIKA
from collections import deque
import time
import matplotlib.pyplot as plt

start_sample=0
no_trial = "trial3"
task = "upper"
path_to_csv_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"
q_path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_mocap_ipopt.csv"

mks_names = settings.marker_mocap_names
#read mks data
df_wide = udp_csv_to_dataframe(path_to_csv_mocap, mks_names)
result_markers_mocap, start_sample_mks_mocap = read_mks_data(df_wide)

path_to_csv_lstm = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/augmented_markers_filtred.csv"
path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/3d_keypoints.csv"
q_path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_cosmik_ipopt.csv"
keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
data_markers_lstm = pd.read_csv(path_to_csv_lstm) 
keypoints = pd.read_csv(path_to_kpt) 
columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
if len(data_markers_lstm) != len(keypoints):
    raise ValueError("Row count mismatch between data_markers_lstm and keypoints")
data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)
result_markers_lstm, start_sample_mks_lstm = read_mks_data(data_markers_lstm, start_sample=start_sample) #check the function of read 


markers_to_display = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study',
           'r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

human_model, human_geom_model, visuals_dict = build_model(start_sample_mks_mocap, meshes_folder_path)

human_model_cosmik, human_geom_model_cosmik, visuals_dict_cosmik = build_model(start_sample_mks_lstm, meshes_folder_path)

# VISUALIZATION
viz = gv_init(human_model,human_geom_model.copy(),human_geom_model)
viz_lstm = GepettoVisualizer(human_model_cosmik,human_geom_model_cosmik.copy(),human_geom_model_cosmik)
viz_lstm.initViewer()
viz_lstm.loadViewerModel("model_cosmik")

#measured frames
# seg_frames = construct_segments_frames(result_markers_mocap[start_sample])
# add_frames(viz,seg_frames,"meas", 0.008, 0.08)
#model markers spheres 
add_marker(viz,start_sample_mks_mocap,"_mocap", 1, 0,0)
add_marker(viz,start_sample_mks_lstm,"_cosmik", 0, 1,0)
#model frames
seg_names_mks = get_segments_mks_dict(result_markers_mocap[start_sample])
seg_names_mks_cosmik = get_segments_mks_dict(result_markers_lstm[start_sample])
# add_frames(viz,seg_names_mks,"model", 0.012, 0.05)


data = human_model.createData()
data_cosmik = human_model_cosmik.createData()
q_mocap = read_joint_angles_wholebody(q_path_mocap, start_sample)
q_cosmik= read_joint_angles_wholebody(q_path_cosmik, start_sample)

for i in range(len(q_mocap)):
    
    pin.forwardKinematics(human_model, data, q_mocap[i])
    pin.updateFramePlacements(human_model, data)

    pin.forwardKinematics(human_model_cosmik, data_cosmik, q_cosmik[i])
    pin.updateFramePlacements(human_model_cosmik, data_cosmik)

    viz.display(q_mocap[i])
    # viz_lstm.display(q_cosmik[i])

        #Display markers from model
    for mk_name in markers_to_display:
        print(mk_name)
        sphere_name_mocap = f'world/{mk_name}_mocap'
        sphere_name_cosmik = f'world/{mk_name}_cosmik'
        mk_position_mocap = data.oMf[human_model.getFrameId(mk_name)].translation
        mk_position_cosmik = data_cosmik.oMf[human_model_cosmik.getFrameId(mk_name)].translation
        place(viz, sphere_name_mocap, pin.SE3(np.eye(3), np.matrix(mk_position_mocap.reshape(3,)).T))
        # place(viz_lstm, sphere_name_cosmik, pin.SE3(np.eye(3), np.matrix(mk_position_cosmik.reshape(3,)).T))

    time.sleep(0.05)

    # input()