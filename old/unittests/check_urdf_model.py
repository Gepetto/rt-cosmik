#This script uses the ii-th frame of the video to construct the model, then,
#displays the pinocchio model in neutral pose with frames, mks, and meshes

ii = 0 #Video frame used for calibration


import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')

import pandas as pd 
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from utils.model_utils import Robot

data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/augmented_markers_positions.csv'))
data_keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))

result_markers = []
for frame, group in data_markers.groupby("Frame"):
    frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    result_markers.append(frame_dict)

result_keypoints = []
for frame, group in data_keypoints.groupby("Frame"):
    frame_dict = {row["Keypoint"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    if "Hip" in frame_dict:
        frame_dict["midHip"] = frame_dict.pop("Hip")
    result_keypoints.append(frame_dict)


lstm_dict = {**result_keypoints[ii], **result_markers[ii]}

human = Robot(os.path.join(rt_cosmik_path,'urdf/human.urdf'),rt_cosmik_path,True,np.array([[1,0,0],[0,0,-1],[0,1,0]])) 
human_model = human.model
human_collision_model = human.collision_model
human_visual_model = human.visual_model

list_index_urdf = [24,25,26,27,28,7,8,14,15,16,17,18,9,10,11,12,13,19,20,21,22,23]
mapping_pin_model_to_urdf_model = dict(zip(np.arange(7,human_model.nq),list_index_urdf))
print(mapping_pin_model_to_urdf_model)

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


#Display model frames, mks, and mesh in neutral pose
q = pin.neutral(human_model) # init pos
data = pin.Data(human_model)
pin.forwardKinematics(human_model, data, q)
pin.updateFramePlacements(human_model, data)
viz.display(q)

input()

q[:] = np.array([[-0.23757297,  1.08821576,  0.19000075, -0.0235671,   0.1025104,  -0.08211563,
  0.99105662,  0.02738802,  0.09087933, -0.13369843, -0.10353136, -0.35257774,
  0.81207362,  1.27408774,  0.0947939 ,  0.10503749, -0.35360598,  0.51428541,
 -1.27408329,  0.08053271, -0.02890399,  0.01561397, -0.0752474,  -0.39091311,
  0.07275684,  0.11844548, -0.02800828,  0.00789671,  0.17154332]])

q_reordered = pin.neutral(human_model)
q_reordered[:7]=q[:7]
for ii in range(7,human_model.nq):
    q_reordered[ii] = q[list_index_urdf.index(ii)]
viz.display(q_reordered)