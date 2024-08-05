import pandas as pd 
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import sys 
import numpy as np
from utils.viz_utils import place, Rquat

model = pin.Model()
visual_model = pin.GeometryModel()
collision_model = pin.GeometryModel()


# Visualisation

viz = GepettoVisualizer(model, visual_model,collision_model)
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

data = pd.read_csv('keypoints_3D_pos_2RGB_2.csv')

for ii in range(1,data['Frame'].iloc[-1]+1): 
    for name in keypoint_names:
        # Filter the DataFrame for the specific frame and keypoint
        keypoint_data = data[(data['Frame'] == ii) & (data['Keypoint'] == name)]
        if not keypoint_data.empty:
            M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([keypoint_data.iloc[0]['X'],keypoint_data.iloc[0]['Y'],keypoint_data.iloc[0]['Z']]).T))
            place(viz,'world/'+keypoint_data.iloc[0]['Keypoint'],M)
    input()
        