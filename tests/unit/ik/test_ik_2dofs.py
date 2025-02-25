import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from utils.read_write_utils import read_mks_data
import pandas as pd
from human_model.urdf_model import * 
from viewer.gv_viewer import place, gv_init
from ik.ik import RT_IK

#read mks data
start_sample = 0
data_markers = pd.read_csv('/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/augmented_data.csv') 

result_markers, start_sample_dict = read_mks_data(data_markers, start_sample=start_sample) #check the function of read 

#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/2dof_human_arm_polishing.urdf',rt_cosmik_path) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))

jcp_dict = get_jcp_global_pos_2dof(start_sample_dict,side_to_track="right")
seg_lengths = calculate_segment_lengths_from_dict_2dof(jcp_dict)
seg_lengths_dic={'Elbow' :seg_lengths[0], 'Wrist' :seg_lengths[1]}

human_model = model_scaling_from_dict_2dof(human_model, seg_lengths_dic)


# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,jcp_dict.keys())

for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

viz.viewer.gui.addSphere('world/elbow_m',0.02,[0,0,1,1])
viz.viewer.gui.addSphere('world/wrist_m',0.02,[0,0,1,1])

q =pin.neutral(human_model)
# q[:]= np.array([0,np.pi/2])
pin.framesForwardKinematics(human_model, human_data, q)
viz.display(q)
input()

#place markers to track
for key in jcp_dict.keys():
    place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))
input()
#place urdf frames
for frame in human_model.frames.tolist():
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
input()
#Print out the placement of each joint of the kinematic tree
for name, oMi in zip(human_model.names, human_data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

input()

### IK calculations 
keys_to_track_list = ['Elbow', 'Wrist']
dict_dof_to_keypoints = dict(zip(['elbow', 'hand_fixed'],keys_to_track_list))

dt = 1/40
ik_class = RT_IK(human_model, jcp_dict, q, keys_to_track_list, dt, dict_dof_to_keypoints, False)



for ii in range(start_sample,len(result_markers)): 
    mks_dict = result_markers[ii]
    jcp_dict = get_jcp_global_pos_2dof(mks_dict,side_to_track="right")

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
    pin.framesForwardKinematics(human_model, human_data, q)
    
    for key in jcp_dict.keys():
        place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

    for frame in human_model.frames.tolist():
        place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

    place(viz, 'world/elbow_m', human_data.oMi[human_model.getJointId('elbow')])
    place(viz, 'world/wrist_m', human_data.oMf[human_model.getFrameId('hand')])

    

    viz.display(q)
    ik_class._q0 = q
    input()