from utils.model_utils import Robot, model_scaling
from utils.read_write_utils import set_zero_data
from utils.viz_utils import Rquat, place
import pandas as pd
import sys
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import numpy as np 

# Loading human urdf
human = Robot('models/human_urdf/urdf/human.urdf','models',True) 
human_model = human.model
human_visual_model = human.visual_model
human_collision_model = human.collision_model

# Loading joint center positions
data = pd.read_csv('output/keypoints_3D_pos_2RGB_2.csv')

set_zero_data(data)
print(data)

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

# Scaling segments lengths 
human_model,human_data=model_scaling(human_model, data[(data['Frame']==1)])

# Visualisation

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
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

viz.display(pin.neutral(human_model))

for keypoint in keypoint_names:
    viz.viewer.gui.addSphere('world/'+keypoint,0.01,color[keypoint])

for ii in range(1,data['Frame'].iloc[-1]+1): 
    for name in keypoint_names:
        # Filter the DataFrame for the specific frame and keypoint
        keypoint_data = data[(data['Frame'] == ii) & (data['Keypoint'] == name)]
        if not keypoint_data.empty:
            M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([keypoint_data.iloc[0]['X'],keypoint_data.iloc[0]['Y'],keypoint_data.iloc[0]['Z']]).T))
            place(viz,'world/'+keypoint_data.iloc[0]['Keypoint'],M)
    input()