import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/old_meshes'
rt_cosmik_path = os.path.dirname(script_directory)
from human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from utils.read_write_utils import read_mks_data
import pandas as pd
from human_model.pin_model import * 
from human_model.model_utils import construct_segments_frames
from viewer.gv_viewer import place, gv_init, Rquat, add_frames

#read mks data
start_sample = 0
data_markers = pd.read_csv('/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/augmented_data.csv') 

result_markers, start_sample_dict = read_mks_data(data_markers, start_sample=start_sample) #check the function of read 

human_model, human_geom_model, visuals_dict = build_model(start_sample_dict,meshes_folder_path)

q =pin.neutral(human_model)
human_data = pin.Data(human_model)
pin.framesForwardKinematics(human_model,human_data,q)
pin.updateFramePlacements(human_model, human_data)


# VISUALIZATION


viz = gv_init(human_model,human_geom_model.copy(),human_geom_model,start_sample_dict.keys())

#display markers
for key in start_sample_dict.keys():
    M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([start_sample_dict[key][0],start_sample_dict[key][1],start_sample_dict[key][2]]).T))
    place(viz,'world/'+key,M)

# construct and display frames 
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)

for seg_name, mks in seg_frames.items():
    frame_name = f'world/{seg_name+"_meas"}'   
    frame_se3 = pin.SE3(mks[:3,:3], np.matrix([mks[0,3],mks[1,3],mks[2,3]]).T)
    place(viz, frame_name, frame_se3)



viz.display(q)