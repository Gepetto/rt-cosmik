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
no_trial = "Maxime"
task = "static"
# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"

path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/mocap/mocap_Maxime/static/mocap_downsampled_to_40hz.csv"
mks_data = pd.read_csv(path_to_csv)/1000
###########################################################################################for cosmik data 
# path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/3d_keypoints_filtered_2.csv"

# keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']

# data_markers_lstm = pd.read_csv(path_to_csv) 
# keypoints = pd.read_csv(path_to_kpt) 

# columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

# mks_data = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

###################################################################""""
start_sample=0
result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample) #check the function of read 




# start_sample=0
# mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
#              'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
#              'BHD','RHD','LHD','FHD',
#              'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
#              'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
#              'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
#              'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
#              'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
#              'r_pelvis', 'l_pelvis']
# # df_raw = pd.read_8data_to_dataframe(df_raw, mks_names) #marker data are string 
# mks_data = udp_csv_to_dataframe(path_to_csv, mks_names) #float
# result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample) #check the function of read 


#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender='male',subject_height=1.81)
print(human_model.nq)

human_model= mks_registration(human_model,start_sample_dict, with_hand=True, gender='male',subject_height=1.81)

human_data = pin.Data(human_model)
human_collision_model = human.collision_model
human_visual_model = human.visual_model

# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,start_sample_dict)
pin.forwardKinematics(human_model,human_data, pin.neutral(human_model))
pin.updateFramePlacements(human_model,human_data)

# display urdf frames

# Print all joint names

get_segment_length(start_sample_dict)
# measured frames
seg_frames = construct_segments_frames(start_sample_dict)
add_frames(viz,seg_frames,"meas", 0.008, 0.08)
for seg_name, M in seg_frames.items():
        
        frame_name = f'world/{seg_name+"_meas"}'
        frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
        place(viz, frame_name, frame_se3)
# #display markers
for key in start_sample_dict.keys():
    place(viz, 'world/'+key, pin.SE3(np.eye(3), np.array([start_sample_dict[key][0],start_sample_dict[key][1],start_sample_dict[key][2]])))

q =pin.neutral(human_model)
# q[13]= -1.745
pin.forwardKinematics(human_model,human_data, q)
pin.updateFramePlacements(human_model,human_data)
for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

viz.display(q)
