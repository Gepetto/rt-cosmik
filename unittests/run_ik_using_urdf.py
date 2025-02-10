
import argparse
import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')
import cv2
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

from collections import deque
from datetime import datetime
import yaml
import time
import pandas as pd
from utils.lstm_v2 import augmentTRC, loadModel
from utils.model_utils import Robot, get_jcp_global_pos, calculate_segment_lengths_from_dict, model_scaling_from_dict, get_jcp_global_pos_2dof, calculate_segment_lengths_from_dict_2dof, model_scaling_from_dict_2dof
from utils.calib_utils import load_cam_params, load_cam_to_cam_params, load_cam_pose, list_cameras_with_v4l2, get_cameras_params
from utils.triangulation_utils import triangulate_points
from utils.ik_utils import RT_IK
from utils.iir import IIR
from utils.viz_utils import visualize, VISUALIZATION_CFG, place, Rquat
from utils.read_write_utils import init_csv, save_3dpos_to_csv, save_q_to_csv, read_mks_data
from utils.settings import Settings
import gepetto as gep
settings = Settings()

data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output//lstm.csv')) 

start_sample=0
##for lstm data 
result_markers = []
# for frame, group in data_markers.groupby("Frame"):
#     frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
#     result_markers.append(frame_dict)

# lstm_dict = result_markers[start_sample]
result_markers, start_sample_dict = read_mks_data(data_markers, start_sample=start_sample) #check the function of read 
lstm_dict = result_markers[start_sample]

#load urdf
human = Robot(os.path.join(rt_cosmik_path,'urdf/2dof_human_arm_polishing.urdf'),rt_cosmik_path) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))
# pos_ankle_calib = human_data.oMi[human_model.getJointId('ankle_Z')].translation


jcp_dict, _ = get_jcp_global_pos_2dof(lstm_dict,settings.side_to_track)
seg_lengths = calculate_segment_lengths_from_dict_2dof(jcp_dict)
# seg_lengths_dic={'Knee' : seg_lengths[0], 'Hip': seg_lengths[1], 'Shoulder' : seg_lengths[2],'Elbow' :seg_lengths[3], 'Wrist' :seg_lengths[4]}
seg_lengths_dic={'Elbow' :seg_lengths[0], 'Wrist' :seg_lengths[1]}
# print(seg_lengths)
human_model = model_scaling_from_dict_2dof(human_model, seg_lengths_dic)


# VISUALIZATION

viz = GepettoVisualizer(human_model,human_collision_model,human_visual_model)
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

viz.viewer.gui.addXYZaxis('world/torso_frame',[1,0,0,1],0.02,0.15)

for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

for key in jcp_dict.keys():
    viz.viewer.gui.addSphere('world/'+key,0.01,[0,0,1,1])
    # viz.viewer.gui.addSphere('world/'+key+"_m",0.1,[0,1,0,1])
viz.viewer.gui.addSphere('world/elbow_m',0.02,[0,1,0,1])
viz.viewer.gui.addSphere('world/wrist_m',0.02,[0,1,0,1])
# viz.viewer.gui.setBackgroundColor1("python-pinocchio", gep.color.Color.white)
# viz.viewer.gui.setBackgroundColor2("python-pinocchio", gep.color.Color.white)
# viz.viewer.gui.addLight("light", "python-pinocchio", 360, gep.color.Color.white)

q =pin.neutral(human_model)
# q[:]= np.array([0,np.pi/2])
viz.display(q)
pin.framesForwardKinematics(human_model, human_data, q)


for key in jcp_dict.keys():
    place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

for frame in human_model.frames.tolist():
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

#Print out the placement of each joint of the kinematic tree
for name, oMi in zip(human_model.names, human_data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

input()

### IK calculations 
# keys_to_track_list = ['Knee', 'midHip', 'Shoulder', 'Elbow', 'Wrist']
# dict_dof_to_keypoints = dict(zip(['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed'],keys_to_track_list))
keys_to_track_list = ['Elbow', 'Wrist']
dict_dof_to_keypoints = dict(zip(['elbow_Z', 'hand_fixed'],keys_to_track_list))

dt = 1/40

ik_class = RT_IK(human_model, jcp_dict, q, keys_to_track_list, dt, dict_dof_to_keypoints, False)



for ii in range(start_sample,len(result_markers)): 
    lstm_dict = result_markers[ii]
    jcp_dict, torso_ppose = get_jcp_global_pos_2dof(lstm_dict,settings.side_to_track)

    theta = -np.pi / 2  
    R_z = np.array([[ np.cos(theta), -np.sin(theta), 0],
                    [ np.sin(theta),  np.cos(theta), 0],
                    [ 0,             0,             1]])
    shoulder_offset = jcp_dict["Shoulder"].copy()
    for key, element in jcp_dict.items():
        jcp_dict[key] = jcp_dict[key] - shoulder_offset
        jcp_dict[key] = np.dot(R_z, jcp_dict[key])

    ik_class._dict_m= jcp_dict
    q = ik_class.solve_ik_sample_casadi() 
    print(q)
    
    place(viz, 'world/torso_frame', pin.SE3(torso_ppose[:3,:3], np.array([torso_ppose[0,3], torso_ppose[1,3], torso_ppose[2,3]])))
    for key in jcp_dict.keys():
        print(key)
        place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

    pin.framesForwardKinematics(human_model, human_data, q)
    place(viz, 'world/elbow_m', human_data.oMi[human_model.getJointId('elbow_Z')])
    place(viz, 'world/wrist_m', human_data.oMf[human_model.getFrameId('hand')])
    for frame in human_model.frames.tolist():
        place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

    viz.display(q)
    ik_class._q0 = q
    input()