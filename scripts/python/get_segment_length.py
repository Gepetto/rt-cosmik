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
import matplotlib.pyplot as plt


mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
               'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']
#read mks data
no_trial = "zoe"
s = "Zoe"
task = "robot_welding" 
base_path = f"/root/workspace/ros_ws/src/rt-cosmik/output"
path = f"{base_path}/mocap_jcp/{no_trial}"
path_to_csv_jcp = f"{path}/{task}_joint_center_positions.csv"

# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/mocap/mocap_{no_trial}/{task}/mocap_downsampled_to_40hz.csv"
subject_mass = 72.0
subject_height = 1.80
gender='male'

start_sample=0
# mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
#              'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
#              'BHD','RHD','LHD','FHD',
#              'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
#              'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
#              'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
#              'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
#              'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
#              'r_pelvis', 'l_pelvis']
# df_wide = pd.read_csv(path_to_csv)
# result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=start_sample,converter = 1000.0) #check the function of read 

df_jcp_mocap = pd.read_csv(path_to_csv_jcp) #jcp mocap
result_jcp_mocap, start_sample_jcp = read_mks_data(df_jcp_mocap, start_sample=start_sample,converter = 1000.0) #check the function of read 

df_jcp = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/cosmik_jcp/{s}/{task}_jcp_hpe_filtered_2.csv") 
# df_jcp = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/cosmik_jcp/{s}/output_{no_trial}.csv") #jcp hpe corrigé avc ml
result_jcp, start_sample_jcp = read_mks_data(df_jcp, start_sample=start_sample,converter = 1.0) #check the function of read 

def compute_lengths_jcp(result_jcp, start_sample):
    """Compute arm segment lengths from joint center positions."""
    norms = {}
    norms['upperarmR'] =np.linalg.norm(result_jcp["RElbow"] - result_jcp["RShoulder"])
    norms['upperarmL'] =np.linalg.norm(result_jcp["LElbow"] - result_jcp["LShoulder"])
    norms['lowerarmR']= np.linalg.norm(result_jcp["RWrist"] - result_jcp["RElbow"])
    norms['lowerarmL']=np.linalg.norm(result_jcp["LWrist"] - result_jcp["LElbow"])

    norms['upperlegR']= np.linalg.norm(result_jcp["RKnee"] - result_jcp["RHip"])
    norms['upperlegL']=np.linalg.norm(result_jcp["LKnee"] - result_jcp["LHip"])
    norms['lowerlegR'] = np.linalg.norm(result_jcp["RAnkle"] - result_jcp["RKnee"])
    norms['lowerlegL'] =np.linalg.norm(result_jcp["LAnkle"] - result_jcp["LKnee"])

    return norms


def compute_lengths_markers(result_markers, start_sample):
    """Compute arm segment lengths from marker positions (averaging left/right markers)."""
    norms = {}
    relbow_center = (result_markers['r_melbow_study'] + result_markers['r_lelbow_study'])/2.0
    lelbow_center = (result_markers['L_melbow_study'] + result_markers['L_lelbow_study'])/2.0
    rwrist_center = (result_markers['r_mwrist_study'] + result_markers['r_lwrist_study'])/2.0
    lwrist_center = (result_markers['L_mwrist_study'] + result_markers['L_lwrist_study'])/2.0

    rhip_center = (result_markers['r.PSIS_study'] + result_markers['r.ASIS_study'])/2.0
    lhip_center = (result_markers['L.PSIS_study'] + result_markers['L.ASIS_study'])/2.0

    rknee_center = (result_markers['r_mknee_study'] + result_markers['r_knee_study'])/2.0
    lknee_center = (result_markers['L_mknee_study'] + result_markers['L_knee_study'])/2.0
    rankle_center = (result_markers['r_mankle_study'] + result_markers['r_ankle_study'])/2.0
    lankle_center = (result_markers['L_mankle_study'] + result_markers['L_ankle_study'])/2.0

    norms['upperarmR'] =np.linalg.norm(relbow_center - result_markers['r_shoulder_study'])
    norms['upperarmL'] =np.linalg.norm(lelbow_center - result_markers['L_shoulder_study'])
    norms['lowerarmR']= np.linalg.norm(rwrist_center - relbow_center)
    norms['lowerarmL']=np.linalg.norm(lwrist_center - lelbow_center)

    norms['upperlegR']= np.linalg.norm(rknee_center - rhip_center)
    norms['upperlegL']=np.linalg.norm(lknee_center - lhip_center)
    norms['lowerlegR'] = np.linalg.norm(rankle_center - rknee_center)
    norms['lowerlegL'] =np.linalg.norm(lankle_center - lknee_center)

    
    return norms

def plot_lengths(result_markers, quid,start_sample=0):
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
        if quid == 'jcp':
            norms_init = compute_lengths_jcp(result_markers[0],start_sample)
            norms = compute_lengths_jcp(result_markers[ii],start_sample)
        else : 
            norms_init = compute_lengths_markers(result_markers[0],start_sample)
            norms = compute_lengths_markers(result_markers[ii],start_sample)

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
        axes[i].set_ylabel("Length")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("Frames")

    plt.tight_layout()
    plt.show()
    
# From joint centers
plot_lengths(result_jcp,'jcp')
# plot_lengths(result_jcp_mocap, 'mks')

def plot_lengths(results_list, labels, quid, start_sample=0):
    """
    Plot segment lengths for multiple datasets in the same subplots.

    results_list: list of result_markers datasets (e.g., [result_jcp, result_jcp_mocap])
    labels: list of labels for each dataset (same length as results_list)
    quid: 'jcp' or 'markers'
    """

    all_norms_list = []
    frames_list = []

    # Precompute norms for all datasets
    for result_markers in results_list:
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
        for ii in range(start_sample, len(result_markers)):
            if quid == 'jcp':
                norms_init = compute_lengths_jcp(result_markers[0], start_sample)
                norms = compute_lengths_jcp(result_markers[ii], start_sample)
            else:
                norms_init = compute_lengths_markers(result_markers[0], start_sample)
                norms = compute_lengths_markers(result_markers[ii], start_sample)

            for key in all_norms.keys():
                all_norms[key].append((norms[key], norms_init[key]))
            frames.append(ii)

        all_norms_list.append(all_norms)
        frames_list.append(frames)

    # --- Plot in subplots ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, key in enumerate(all_norms_list[0].keys()):
        for dataset_idx, all_norms in enumerate(all_norms_list):
            values = [v[0] for v in all_norms[key]]
            norm_init = all_norms[key][0][1]
            line, = axes[i].plot(frames_list[dataset_idx], values, label=f"{labels[dataset_idx]}")
            color = line.get_color()
            axes[i].axhline(norm_init,color = color, linestyle='--')
        axes[i].set_title(key)
        axes[i].set_ylabel("Length")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("Frames")

    plt.tight_layout()
    plt.show()


# ✅ Example usage:
plot_lengths(
    [result_jcp, result_jcp_mocap],
    labels=["HPE_JCP", "MoCap_JCP"],
    quid="jcp"
)

# From markers
# Rupper_mks, Lupper_mks, Rlower_mks, Llower_mks = compute_lengths_markers(result_markers, start_sample)
# plot_lengths(time, Rupper_mks, Lupper_mks, Rlower_mks, Llower_mks, title_prefix="Markers")

# all_norms = {
#     'upperlegR': [],
#     'lowerlegR': [],
#     'upperlegL': [],
#     'lowerlegL': [],
#     'upperarmR': [],
#     'lowerarmR': [],
#     'upperarmL': [],
#     'lowerarmL': []
# }

# frames = []

# for ii in range(start_sample,len(result_markers)): 
#     norms_init = get_segment_length(result_markers[0])
#     norms = get_segment_length(result_markers[ii])
#     for key in all_norms.keys():
#         all_norms[key].append(norms[key])
#     frames.append(ii)

# # --- Plot en subplots ---
# # --- Plot avec un subplot par segment ---
# fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
# axes = axes.flatten()

# for i, (key, values) in enumerate(all_norms.items()):
#     axes[i].plot(frames, values, label="Current length")
#     axes[i].axhline(norms_init[key], color='r', linestyle='--', label="Initial length")
#     axes[i].set_title(key)
#     axes[i].set_ylabel("Length (mm)")
#     axes[i].grid(True)
#     axes[i].legend()

# axes[-1].set_xlabel("Frames")

# plt.tight_layout()
# plt.show()
