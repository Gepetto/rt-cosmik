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
from utils.model_utils import build_model_challenge
from utils.ik_utils import RT_SWIKA_FATROP
from utils.viz_utils import place, Rquat
from collections import deque

# parameters
dt = 0.04
T=10

# Loading of the data and the pinocchio model
deque_lstm_dict = deque(maxlen=10)

data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/saved/augmented_markers_positions.csv'))
data_keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/saved/keypoints_3d_positions.csv'))

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

lstm_dict = {**result_keypoints[0], **result_markers[0]}
human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)
human_data = human_model.createData()

keys_to_track_list = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

x_list = []
u_list = []

x0 = np.zeros(human_model.nq + human_model.nv)
x0[6] = 1

for ii in range(T):
    x_list.append(x0)
    u_list.append(np.zeros(human_model.nv))
    deque_lstm_dict.append(lstm_dict)

x_array = np.array(x_list).T
u_array = np.array(u_list).T

lstm_dict_list = list(deque_lstm_dict)

# Convert the list of dictionaries to a NumPy array
array_data = np.array([np.hstack([d[marker] for marker in keys_to_track_list]) for d in lstm_dict_list]).T

cost_weights = np.array([1, 1e-3, 1e-5])

### IK calculations
ik_class = RT_SWIKA_FATROP(human_model, keys_to_track_list, T)
X, U = ik_class.solve(x_array, u_array, array_data, x0, cost_weights, dt)

print(X,U)

# # VISUALIZATION

# viz = GepettoVisualizer(human_model,human_geom_model.copy(),human_geom_model)
# try:
#     viz.initViewer()
# except ImportError as err:
#     print(
#         "Error while initializing the viewer. It seems you should install gepetto-viewer"
#     )
#     print(err)
#     sys.exit(0)

# try:
#     viz.loadViewerModel("pinocchio")
# except AttributeError as err:
#     print(
#         "Error while loading the viewer model. It seems you should start gepetto-viewer"
#     )
#     print(err)
#     sys.exit(0)

# dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

# model_frames=human_model.frames.tolist()
# for dof in dof_names:
#     viz.viewer.gui.addXYZaxis('world/'+dof,[1,0,0,1],0.01,0.1)

# #Blue markers
# for marker in result_markers[0].keys():
#     viz.viewer.gui.addSphere('world/estimated'+marker,0.01,[1,0,0,1])
#     viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
#     M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[0][marker][0],result_markers[0][marker][1],result_markers[0][marker][2]]).T))
#     place(viz,'world/'+marker,M)

# marker_names = keys_to_track_list

# for ii in range(X.shape[1]):
#     q = X[:human_model.nq,ii]
#     viz.display(q)
#     pin.forwardKinematics(human_model, human_data,q)
#     pin.updateFramePlacements(human_model, human_data)
#     for dof in dof_names:
#         place(viz,'world/'+dof,human_data.oMi[human_model.getJointId(dof)])

#     #Red estimated markers
#     for marker in result_markers[0].keys():
#         M = human_data.oMf[human_model.getFrameId(marker)]
#         place(viz,'world/estimated'+marker,M)
#     input()

# SECOND WINDOW AFTER INIT 

# Redefinition of parameters 
x0 = X[:,-1]
lstm_dict = {**result_keypoints[1], **result_markers[1]}
deque_lstm_dict.append(lstm_dict)

lstm_dict_list = list(deque_lstm_dict)
array_data = np.array([np.hstack([d[marker] for marker in keys_to_track_list]) for d in lstm_dict_list]).T

X, U = ik_class.solve(X, U, array_data, x0, cost_weights, dt)
print(X,U)