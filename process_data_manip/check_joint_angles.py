import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/'
rt_cosmik_path = os.path.dirname(script_directory)
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody
import pandas as pd
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.pin_model import build_model
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from src.rtcosmik.ik.ik import RT_IK,RT_SWIKA
from collections import deque
import time
import matplotlib.pyplot as plt

start_sample=0
no_trial = "trial_2"
task = "trial_static"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_pose.csv"
q_path= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mocap_on_cosmik_frames.csv"

mks_names = settings.marker_mocap_names
#read mks data
df_raw = pd.read_csv(path_to_csv)  # original with 'marker_data'
df_wide = marker_data_to_dataframe(df_raw, mks_names)
result_markers, start_sample_mks = read_mks_data(df_wide)

# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/markers.csv"
# path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/keypoints.csv"
# q_path= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/joint_angles.csv"
# keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
# data_markers_lstm = pd.read_csv(path_to_csv) 
# keypoints = pd.read_csv(path_to_kpt) 
# columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]
# if len(data_markers_lstm) != len(keypoints):
#     raise ValueError("Row count mismatch between data_markers_lstm and keypoints")
# data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)
# result_markers, start_sample_mks = read_mks_data(data_markers_lstm, start_sample=start_sample) #check the function of read 


human_model, human_geom_model, visuals_dict = build_model(start_sample_mks, meshes_folder_path)

# VISUALIZATION
viz = gv_init(human_model,human_geom_model.copy(),human_geom_model)
#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)
#model markers spheres 
# add_marker(viz,result_markers[1].keys(), 0, 1,0)
#model frames
seg_names_mks = get_segments_mks_dict(result_markers[start_sample])
add_frames(viz,seg_names_mks,"model", 0.012, 0.05)


data = human_model.createData()
print("ok")
q = read_joint_angles_wholebody(q_path, start_sample)
print("ik")
for seg_name, mks in seg_names_mks.items():
    viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [255, 0., 0, 1.], 0.008, 0.08)
    for mk_name in mks:
            sphere_name_mocap = f'world/{mk_name}_mocap'
            sphere_name_cosmik = f'world/{mk_name}_cosmik'
            # print(sphere_name_cosmik)
            viz.viewer.gui.addSphere(sphere_name_mocap, 0.01, [255, 0., 0, 1.])
for i in range(len(q)):
    
    # print(q[i])
    pin.forwardKinematics(human_model, data, q[i])
    pin.updateFramePlacements(human_model, data)

    viz.display(q[i])
    print(q[i])

    #Display frames from human_model
    for seg_name, mks in seg_names_mks.items():
        
        frame_name = f'world/{seg_name}'
        frame_se3= data.oMf[human_model.getFrameId(seg_name)]
        place(viz, frame_name, frame_se3)


    for seg_name, mks in seg_names_mks.items():
        #Display markers from model
            for mk_name in mks:
                sphere_name_mocap = f'world/{mk_name}_mocap'
                sphere_name_cosmik = f'world/{mk_name}_cosmik'
                mk_position_mocap = data.oMf[human_model.getFrameId(mk_name)].translation
                place(viz, sphere_name_mocap, pin.SE3(np.eye(3), np.matrix(mk_position_mocap.reshape(3,)).T))
    time.sleep(0.05)
    # input()
names  = settings.joint_angles_names
output_dir = "joint_angle_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot one figure per joint
for i, name in enumerate(names):
    plt.figure()
    plt.plot(q[:, i])
    plt.title(name)
    plt.xlabel("Frame")
    plt.ylabel("Angle (rad or deg)")
    plt.grid(True)
    plt.tight_layout()
    
    # Optional: save each plot
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.show()  # Close the figure to avoid too many open windows

print(f"All plots saved in: {output_dir}")