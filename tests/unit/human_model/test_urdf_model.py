import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data,udp_csv_to_dataframe,marker_data_to_dataframe
import pandas as pd
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.viewer.gv_viewer import place, gv_init,add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length

#read mks data
no_trial = "trial_2"
task = "trial_static"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/markers.csv"
###########################################################################################for cosmik data 
path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/keypoints.csv"

keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']

data_markers_lstm = pd.read_csv(path_to_csv) 
keypoints = pd.read_csv(path_to_kpt) 

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

if len(data_markers_lstm) != len(keypoints):
    raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

mks_data = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

###################################################################""""
start_sample=0
mks_names = settings.marker_mocap_names
# df_raw = pd.read_csv(path_to_csv)  # 
# mks_data = marker_data_to_dataframe(df_raw, mks_names) #marker data are string 
# mks_data = udp_csv_to_dataframe(path_to_csv, mks_names) #float
result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample) #check the function of read 

#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True)
print(human_model.nq)

human_model= mks_registration(human_model,start_sample_dict, with_hand=True)

human_data = pin.Data(human_model)
# human_collision_model = human.collision_model
# human_visual_model = human.visual_model

# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,start_sample_dict)
pin.forwardKinematics(human_model,human_data, pin.neutral(human_model))
pin.updateFramePlacements(human_model,human_data)

# display urdf frames
for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

# get_segment_length(start_sample_dict)
# measured frames
# seg_frames = construct_segments_frames(start_sample_dict)
# add_frames(viz,seg_frames,"meas", 0.008, 0.08)
# for seg_name, M in seg_frames.items():
        
#         frame_name = f'world/{seg_name+"_meas"}'
#         frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
#         place(viz, frame_name, frame_se3)
# # # #display markers
# for key in start_sample_dict.keys():
    # place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([start_sample_dict[key][0],start_sample_dict[key][1],start_sample_dict[key][2]])))

q =pin.neutral(human_model)

viz.display(q)
