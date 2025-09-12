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
no_trial = "4279"
task = "robot_welding" #hitting sat probleme
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mocap_downsampled_to_40hz.csv"
path_to_csv_jcp =  f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/joint_center_positions_test.csv"
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
df_wide = pd.read_csv(path_to_csv)
result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=start_sample,converter = 10.0) #check the function of read 

df_jcp = pd.read_csv(path_to_csv_jcp)
result_jcp, start_sample_jcp = read_mks_data(df_jcp, start_sample=start_sample,converter = 10.0) #check the function of read 

def compute_lengths_jcp(result_jcp, start_sample):
    """Compute arm segment lengths from joint center positions."""
    Rupper, Lupper, Rlower, Llower = [], [], [], []
    for j in range(start_sample, len(result_jcp)):
        Rupper.append(np.linalg.norm(result_jcp[j]["RElbow"] - result_jcp[j]["RShoulder"]))
        Lupper.append(np.linalg.norm(result_jcp[j]["LElbow"] - result_jcp[j]["LShoulder"]))
        Rlower.append(np.linalg.norm(result_jcp[j]["RWrist"] - result_jcp[j]["RElbow"]))
        Llower.append(np.linalg.norm(result_jcp[j]["LWrist"] - result_jcp[j]["LElbow"]))
    return Rupper, Lupper, Rlower, Llower


def compute_lengths_markers(result_markers, start_sample):
    """Compute arm segment lengths from marker positions (averaging left/right markers)."""
    Rupper, Lupper, Rlower, Llower = [], [], [], []
    for ii in range(start_sample, len(result_markers)):
        relbow_center = (result_markers[ii]['r_melbow_study'] + result_markers[ii]['r_lelbow_study'])/2.0
        lelbow_center = (result_markers[ii]['L_melbow_study'] + result_markers[ii]['L_lelbow_study'])/2.0
        rwrist_center = (result_markers[ii]['r_mwrist_study'] + result_markers[ii]['r_lwrist_study'])/2.0
        lwrist_center = (result_markers[ii]['L_mwrist_study'] + result_markers[ii]['L_lwrist_study'])/2.0

        Rupper.append(np.linalg.norm(relbow_center - result_markers[ii]['r_shoulder_study']))
        Lupper.append(np.linalg.norm(lelbow_center - result_markers[ii]['L_shoulder_study']))
        # Rlower.append(np.linalg.norm(result_markers[ii]['SJN'] - result_markers[ii]['C7_study']))

        Rlower.append(np.linalg.norm(rwrist_center -relbow_center ))
        Llower.append(np.linalg.norm(lwrist_center-lelbow_center))
    return Rupper, Lupper, Rlower, Llower

def plot_lengths(time, Rupper, Lupper, Rlower, Llower, title_prefix=""):
    """Plot lengths in 4 subplots for comparison."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axs[0, 0].plot(time, Rupper, label="Right Upper Arm", color="r")
    axs[0, 0].set_title(f"{title_prefix} Right Upper Arm"); axs[0, 0].legend()

    axs[0, 1].plot(time, Lupper, label="Left Upper Arm", color="b")
    axs[0, 1].set_title(f"{title_prefix} Left Upper Arm"); axs[0, 1].legend()

    axs[1, 0].plot(time, Rlower, label="Right Lower Arm", color="g")
    axs[1, 0].set_title(f"{title_prefix} Right Lower Arm"); axs[1, 0].legend()

    axs[1, 1].plot(time, Llower, label="Left Lower Arm", color="m")
    axs[1, 1].set_title(f"{title_prefix} Left Lower Arm"); axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_ylabel("Length")  # or meters if scaled
        ax.grid(True)

    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# === Example usage ===
time = np.arange(start_sample, len(result_markers))

# From joint centers
Rupper_jcp, Lupper_jcp, Rlower_jcp, Llower_jcp = compute_lengths_jcp(result_jcp, start_sample)
plot_lengths(time, Rupper_jcp, Lupper_jcp, Rlower_jcp, Llower_jcp, title_prefix="JCP")

# From markers
Rupper_mks, Lupper_mks, Rlower_mks, Llower_mks = compute_lengths_markers(result_markers, start_sample)
plot_lengths(time, Rupper_mks, Lupper_mks, Rlower_mks, Llower_mks, title_prefix="Markers")

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
