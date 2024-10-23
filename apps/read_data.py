import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))

import pandas as pd 
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import sys 
import numpy as np
from utils.viz_utils import place, Rquat
from utils.model_utils import Robot
import matplotlib.pyplot as plt 
import time

# Loading human urdf
human = Robot(os.path.join(rt_cosmik_path,'models/human_urdf/urdf/human.urdf'),os.path.join(rt_cosmik_path,'models')) 
human_model = human.model
human_data = human_model.createData()
human_visual_model = human.visual_model
human_collision_model = human.collision_model

# Visualisation

viz = GepettoVisualizer(human_model, human_collision_model,human_visual_model)
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
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

# Define the keypoint names in order as per the Halpe 26-keypoint format
# keypoint_names = [ "Right Shoulder", "Right Elbow", "Right Wrist", "Right Hip", 
#     "Right Knee", "Right Ankle","Right Big Toe", "Right Small Toe", "Right Heel"
# ]

keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]


orange = [1,165/255,0,1]
blue = [0,0,1,1]
green = [0,1,0,1]

color_values=[blue,blue,blue,blue,blue,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange]

color = dict(zip(keypoint_names,color_values))

for keypoint in keypoint_names:
    viz.viewer.gui.addSphere('world/'+keypoint,0.01,color[keypoint])

data = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))
q=pd.read_csv(os.path.join(rt_cosmik_path,'output/q.csv')).to_numpy()[:,-5:].astype(float)

# # Convert X, Y, Z columns to a numpy array
# keypoints = data[['X', 'Y', 'Z']].values  # Shape: (3, N) where N is the number of keypoints

# # Loading camera pose 
# R1_global, T1_global = load_cam_pose('cams_calibration/cam_params/camera1_pose_test_test.yml')
# R1_global = R1_global # aligns measurements to human model definition

# # Subtract the translation vector (shifting the origin)
# keypoints_shifted = keypoints +T1_global.T

# # Apply the rotation matrix to align the points
# keypoints_transformed = np.dot(keypoints_shifted,R1_global.T)

# # keypoints_transformed-=T1_global.T

# # Update the DataFrame with the transformed points, replacing the old X, Y, Z values
# data['X'], data['Y'], data['Z'] = keypoints_transformed[:, 0], keypoints_transformed[:, 1], keypoints_transformed[:, 2]

# hand_frame_tr = np.zeros((601,3))
c=0

input()

for ii in range(1,data['Frame'].iloc[-1]+1): 
    if 200 <= ii<= 1000: 
        for name in keypoint_names:
            # Filter the DataFrame for the specific frame and keypoint
            keypoint_data = data[(data['Frame'] == ii) & (data['Keypoint'] == name)]
            if not keypoint_data.empty:
                M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([keypoint_data.iloc[0]['X'],keypoint_data.iloc[0]['Y'],keypoint_data.iloc[0]['Z']]).T))
                place(viz,'world/'+keypoint_data.iloc[0]['Keypoint'],M)
        q_ii = q[ii,:]
        viz.display(q_ii)
        pin.forwardKinematics(human_model,human_data,q_ii)
        pin.updateFramePlacements(human_model,human_data)
        # hand_frame_tr[c,:]=human_data.oMf[human_model.getFrameId('hand')].translation
        c+=1
        time.sleep(0.033)
        

# fig, ax = plt.subplots(3,1)
# ax[0].plot(hand_frame_tr[:,0])
# ax[0].set_ylabel('x')
# ax[1].plot(hand_frame_tr[:,1])
# ax[1].set_ylabel('y')
# ax[2].plot(hand_frame_tr[:,2])
# ax[2].set_ylabel('z')
# plt.title('Hand translation')
# plt.show()