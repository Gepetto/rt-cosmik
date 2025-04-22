import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')

import pandas as pd 
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer, RVizVisualizer
import numpy as np
from utils.model_w_mocap_utils import build_model_challenge, get_segments_lstm_mks_dict_challenge, get_subset_challenge_mks_names, construct_segments_frames_challenge
from utils.ik_utils import RT_IK
import time
import csv
from utils.read_write_utils import read_mks_data, read_joint_angles_wholebody
from utils.settings import Settings
from utils.iir import IIR
from utils.viz_utils import place, Rquat, visualize_model_and_measurements


q_cosmik_path = 'q/q_cosmik_qp_interpolated_33Hz.csv'
lstm_mks = pd.read_csv(os.path.join(rt_cosmik_path,'process_data_manip/mks_lstm/augmented_markers_positions_test_2.csv')) #path to mks data

q_mocap_path = 'q/q_mocap_qp_downsampled_33Hz.csv'
mocap_mks = pd.read_csv(os.path.join(rt_cosmik_path,'process_data_manip/mks_mocap/mks_mocap_test_2.csv')) #path to mks data


start_sample=0
##for lstm data 
result_markers_lstm = []
for frame, group in lstm_mks.groupby("Frame"):
    frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    result_markers_lstm.append(frame_dict)

lstm_dict = result_markers_lstm[start_sample]
# model_lstm, geom_model_lstm, visuals_dict_lstm = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)


# for mocap data
result_markers, mocap_dict = read_mks_data(mocap_mks, start_sample=start_sample) #check the function of read 

mocap_dict = result_markers[start_sample]
human_model, human_geom_model, visuals_dict = build_model_challenge(mocap_dict, mocap_dict, meshes_folder_path)

model_lstm, geom_model_lstm, visuals_dict_lstm = build_model_challenge(mocap_dict, mocap_dict, meshes_folder_path)


# VISUALIZATION

viz = GepettoVisualizer(human_model,human_geom_model.copy(),human_geom_model)

try:
    viz.initViewer()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install gepetto-viewer"
    )
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("model_mocap")
except AttributeError as err:
    print(
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

viz_lstm = GepettoVisualizer(model_lstm,geom_model_lstm.copy(),geom_model_lstm)
viz_lstm.initViewer()
viz_lstm.loadViewerModel("model_cosmik")


seg_names_mks = get_segments_lstm_mks_dict_challenge()

for name, visual in visuals_dict.items():
    viz.viewer.gui.setColor(viz.getViewerNodeName(visual, pin.GeometryType.VISUAL), [1, 0, 0, 0.5])

for name, visual in visuals_dict_lstm.items():
    viz_lstm.viewer.gui.setColor(viz_lstm.getViewerNodeName(visual, pin.GeometryType.VISUAL), [0, 1, 0, 0.5])



data_lstm = model_lstm.createData()
q_cosmik = read_joint_angles_wholebody(q_cosmik_path, 0, 0)


data = human_model.createData()
q_mocap = read_joint_angles_wholebody(q_mocap_path, 35, 34)

for seg_name, mks in seg_names_mks.items():
    viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [255, 0., 0, 1.], 0.008, 0.08)
    for mk_name in mks:
            sphere_name_mocap = f'world/{mk_name}_mocap'
            sphere_name_cosmik = f'world/{mk_name}_cosmik'
            print(sphere_name_cosmik)
            viz.viewer.gui.addSphere(sphere_name_mocap, 0.01, [255, 0., 0, 1.])
            viz_lstm.viewer.gui.addSphere(sphere_name_cosmik, 0.01, [0, 255., 0, 1.])
input()
for i in range(len(q_cosmik)):
    
    print(q_mocap[i])
    pin.forwardKinematics(human_model, data, q_mocap[i])
    pin.updateFramePlacements(human_model, data)

    pin.forwardKinematics(model_lstm, data_lstm, q_cosmik[i])
    pin.updateFramePlacements(model_lstm, data_lstm)

    viz.display(q_mocap[i])
    viz_lstm.display(q_cosmik[i])

    for seg_name, mks in seg_names_mks.items():
        #Display markers from model
            for mk_name in mks:
                sphere_name_mocap = f'world/{mk_name}_mocap'
                sphere_name_cosmik = f'world/{mk_name}_cosmik'
                mk_position_mocap = data.oMf[human_model.getFrameId(mk_name)].translation
                mk_position_cosmik = data_lstm.oMf[model_lstm.getFrameId(mk_name)].translation
                place(viz, sphere_name_mocap, pin.SE3(np.eye(3), np.matrix(mk_position_mocap.reshape(3,)).T))
                place(viz_lstm, sphere_name_cosmik, pin.SE3(np.eye(3), np.matrix(mk_position_cosmik.reshape(3,)).T))
    time.sleep(0.01)
    # input()
