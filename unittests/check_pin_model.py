#This script uses the ii-th frame of the video to construct the model, then,
#displays the pinocchio model in neutral pose with frames, mks, and meshes

ii = 0 #Video frame used for calibration


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
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from utils.model_utils import build_model_challenge, get_segments_lstm_mks_dict_challenge, get_subset_challenge_mks_names
from utils.viz_utils import place
import time

data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/augmented_markers_positions.csv'))
data_keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))

result_markers = []
for frame, group in data_markers.groupby("Frame"):
    frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    result_markers.append(frame_dict)

result_keypoints = []
for frame, group in data_keypoints.groupby("Frame"):
    frame_dict = {row["Keypoint"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    if "Hip" in frame_dict:
        frame_dict["midHip"] = frame_dict.pop("Hip")
    result_keypoints.append(frame_dict)


lstm_dict = {**result_keypoints[ii], **result_markers[ii]}

t1 =time.time()
human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)
t2 = time.time()

print("Time to build the human_model: ", t2-t1)

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
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print(
        "Error while loading the viewer human_model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

seg_names_mks = get_segments_lstm_mks_dict_challenge()
mks_names = get_subset_challenge_mks_names()

#Display model frames, mks, and mesh in neutral pose
q = pin.neutral(human_model) # init pos
data = pin.Data(human_model)
pin.forwardKinematics(human_model, data, q)
pin.updateFramePlacements(human_model, data)
viz.display(q)
for seg_name, mks in seg_names_mks.items():
    #Display markers from human_model
    for mk_name in mks:
        mk_name_gui = f'world/{mk_name}'
        viz.viewer.gui.addSphere(mk_name_gui,0.01,[0,0,1,1])
        mk_position = data.oMf[human_model.getFrameId(mk_name)].translation
        place(viz, mk_name_gui, pin.SE3(np.eye(3), np.matrix(mk_position.reshape(3,)).T))
    
    #Display frames from human_model
    frame_name = f'world/{seg_name}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.008, 0.08)
    frame_se3= data.oMf[human_model.getFrameId(seg_name)]
    place(viz, frame_name, frame_se3)
