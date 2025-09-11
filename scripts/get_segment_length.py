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
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length
from src.rtcosmik.ik.ik import RT_IK


mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
               'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']
#read mks data
no_trial = "4279"
task = "robot_welding" #hitting sat probleme
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mocap_downsampled_to_40hz.csv"

subject_mass = 72.0
subject_height = 1.80
gender='male'

start_sample=0
mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']
# df_raw = pd.read_data_to_dataframe(df_raw, mks_names) #marker data are string 
df_wide = pd.read_csv(path_to_csv)
# mks_data = udp_csv_to_dataframe(path_to_csv, mks_names) #float
result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=start_sample,converter = 1000.0) #check the function of read 
# print(result_markers)
# input()
#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender=gender,subject_height=subject_height)
human_model= mks_registration(human_model,start_sample_dict, with_hand=True)
human_data = pin.Data(human_model)
print(human_model.nq)

################################################################################LOCK JOINTS
all_joint_ids = set(range(1, human_model.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if human_model.existJointName(jn):
        joint_ids_to_lock.append(human_model.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(human_model)
# Build reduced model
human_model, human_visual_model = pin.buildReducedModel(
    human_model, human_visual_model, joint_ids_to_lock, q0)

print(human_model.nq)
human_data = pin.Data(human_model)
###############################################################################################################
# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,start_sample_dict)
pin.forwardKinematics(human_model,human_data, pin.neutral(human_model))
pin.updateFramePlacements(human_model,human_data)

# display urdf frames
for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
    
q =pin.neutral(human_model)
viz.display(q)
input("model scaled, you can launch ik")

#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)

#model markers spheres 
add_marker(viz,result_markers[1].keys(),'_m', 0, 1,0)
#model frames
for joint_id in range(1, human_model.njoints):  # Skip 0 (universe)
    frame_name = f'world/{human_model.names[joint_id]+"_model"}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.012, 0.05)



### IK init 
q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)

dt = 1/40 #dt for qp
#track only real markers (without technical markers)
keys_to_track_list = [
        'BHD','RHD','LHD','FHD',
        'C7_study',
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study'
    ]


rmse_per_marker = {}
q_list = []
M_model_list = []

import matplotlib.pyplot as plt

all_norms = {
    'upperlegR': [],
    'lowerlegR': [],
    'upperlegL': [],
    'lowerlegL': [],
    'upperarmR': [],
    'lowerarmR': [],
    'upperarmL': [],
    'lowerarmL': []
}

frames = []

for ii in range(start_sample,len(result_markers)): 
    norms_init = get_segment_length(result_markers[0])
    norms = get_segment_length(result_markers[ii])
    for key in all_norms.keys():
        all_norms[key].append(norms[key])
    frames.append(ii)

# --- Plot en subplots ---
# --- Plot avec un subplot par segment ---
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for i, (key, values) in enumerate(all_norms.items()):
    axes[i].plot(frames, values, label="Current length")
    axes[i].axhline(norms_init[key], color='r', linestyle='--', label="Initial length")
    axes[i].set_title(key)
    axes[i].set_ylabel("Length (mm)")
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel("Frames")

plt.tight_layout()
plt.show()
