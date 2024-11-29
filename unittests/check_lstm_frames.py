#This script uses the ii-th frame of the video to construct the model frames from lstm mks, then,
#displays the frames and mks for the calibration posture

#In gepetto vizualisation, blue mks correspond to lstm mks, green ones correspond to mmpose mks.

ii = 100 #Video frame used for calibration

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
from utils.model_utils import build_model_challenge, construct_segments_frames_challenge
from utils.viz_utils import place, Rquat
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

print("Time to build the model: ", t2-t1)

# VISUALIZATION

viz = GepettoVisualizer(human_model)
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
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

# for ii in range(len(result_markers)):
#     print(ii)

sgts_poses = construct_segments_frames_challenge(lstm_dict)

#Blue markers
for marker in result_markers[ii].keys():
    viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
    M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
    place(viz,'world/'+marker,M)
    # input("Press Enter to continue...")

for marker in result_keypoints[ii].keys():
    viz.viewer.gui.addSphere('world/'+marker,0.01,[0,1,0,1])
    M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_keypoints[ii][marker][0],result_keypoints[ii][marker][1],result_keypoints[ii][marker][2]]).T))
    place(viz,'world/'+marker,M)

for frame in sgts_poses.keys():
    viz.viewer.gui.addXYZaxis('world/'+frame, [255, 0., 0, 1.], 0.008, 0.08)
    place(viz,'world/'+frame,pin.SE3(sgts_poses[frame][:3,:3], np.matrix(sgts_poses[frame][:3,3].reshape(3,)).T))

