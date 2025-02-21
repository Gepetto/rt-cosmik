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

#read mks data
start_sample = 0
data_markers = pd.read_csv('/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/lstm_frontal_plan.csv') 

result_markers, start_sample_dict = read_mks_data(data_markers, start_sample=start_sample) #check the function of read 

#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/2dof_human_arm_polishing.urdf',rt_cosmik_path) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))

jcp_dict, _ = get_jcp_global_pos_2dof(start_sample_dict,side_to_track="right")
seg_lengths = calculate_segment_lengths_from_dict_2dof(jcp_dict)
seg_lengths_dic={'Elbow' :seg_lengths[0], 'Wrist' :seg_lengths[1]}

human_model = model_scaling_from_dict_2dof(human_model, seg_lengths_dic)


# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,jcp_dict.keys())
#displa urdf frames
for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

#display markers
for key in jcp_dict.keys():
    place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([jcp_dict[key][0],jcp_dict[key][1],jcp_dict[key][2]])))

q =pin.neutral(human_model)

viz.display(q)
