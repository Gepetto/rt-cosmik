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
from utils.ik_utils import RT_SWIKA, RT_IK
from utils.viz_utils import place, Rquat
import time
from collections import deque
import csv

ii=0
q_csv_file_path = os.path.join(rt_cosmik_path,'output/q_swika.csv')

dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

with open(q_csv_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write the header row
    csv_writer.writerow(['FF_X', 'FF_Y', 'FF_Z', 'FF_QUAT_X', 'FF_QUAT_Y', 'FF_QUAT_Z','FF_QUAT_W' ] + dof_names)

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

lstm_dict = {**result_keypoints[ii], **result_markers[ii]}

t1 =time.time()
human_model, human_geom_model, visuals_dict = build_model_challenge(lstm_dict, lstm_dict, meshes_folder_path)
t2 = time.time()
print("Time to build the model: ", t2-t1)

human_data = human_model.createData()

# VISUALIZATION

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

dof_names=['middle_lumbar_Z', 'middle_lumbar_Y', 'right_shoulder_Z', 'right_shoulder_X', 'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y', 'left_elbow_Z', 'left_elbow_Y', 'right_hip_Z', 'right_hip_X', 'right_hip_Y', 'right_knee_Z', 'right_ankle_Z','left_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_knee_Z', 'left_ankle_Z'] 

model_frames=human_model.frames.tolist()
# for dof in dof_names:
#     viz.viewer.gui.addXYZaxis('world/'+dof,[1,0,0,1],0.01,0.1)

marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

### IK init 
dt = 1/10
T=10

keys_to_track_list = marker_names

x_list = []

x0 = np.zeros(human_model.nq + human_model.nv)
x0[6] = 1

for ii in range(T):
    x_list.append(x0)
    deque_lstm_dict.append(lstm_dict)

### IK calculations
ik_class = RT_SWIKA(human_model, deque_lstm_dict, x_list, keys_to_track_list, T, dt)
sol, new_x_list = ik_class.solve_swika_casadi()
q = new_x_list[-1][:human_model.nq]

# viz.display(q)
# Saving kinematics
with open(q_csv_file_path, mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    # Write to CSV
    csv_writer.writerow(q.tolist())


# pin.forwardKinematics(human_model, human_data,q)
# pin.updateFramePlacements(human_model, human_data)
# for dof in dof_names:
#     place(viz,'world/'+dof,human_data.oMi[human_model.getJointId(dof)])

# #Blue markers
# for marker in result_markers[ii].keys():
#     viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
#     M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
#     place(viz,'world/'+marker,M)
#     # input("Press Enter to continue...")

# #Red estimated markers
# for marker in result_markers[ii].keys():
#     viz.viewer.gui.addSphere('world/estimated'+marker,0.01,[1,0,0,1])
#     M = human_data.oMf[human_model.getFrameId(marker)]
#     place(viz,'world/estimated'+marker,M)


while ii < len(result_markers):
    ii+=1
    print(ii)

    lstm_dict = {**result_keypoints[ii], **result_markers[ii]}
    deque_lstm_dict.append(lstm_dict)

    ik_class._x_list = new_x_list
    ik_class._deque_dict_m = deque_lstm_dict

    sol, new_x_list = ik_class.solve_swika_casadi()

    q = new_x_list[-1][:human_model.nq]

    # viz.display(q)
    # Saving kinematics
    with open(q_csv_file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        # Write to CSV
        csv_writer.writerow(q.tolist())

    # pin.forwardKinematics(human_model, human_data, q)
    # pin.updateFramePlacements(human_model, human_data)
    # for dof in dof_names:
    #     place(viz,'world/'+dof,human_data.oMi[human_model.getJointId(dof)])

    # #Blue markers
    # for marker in result_markers[ii].keys():
    #     viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
    #     M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
    #     place(viz,'world/'+marker,M)
    #     # input("Press Enter to continue...")

    # #Red estimated markers
    # for marker in result_markers[ii].keys():
    #     viz.viewer.gui.addSphere('world/estimated'+marker,0.01,[1,0,0,1])
    #     M = human_data.oMf[human_model.getFrameId(marker)]
    #     place(viz,'world/estimated'+marker,M)

    # input()
