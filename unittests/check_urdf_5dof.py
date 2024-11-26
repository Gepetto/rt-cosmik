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
from utils.viz_utils import place

human = Robot(os.path.join(rt_cosmik_path,'urdf/human_5dof.urdf'),rt_cosmik_path) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

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


for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

q0 =pin.neutral(human_model)
q0[:]= np.array([np.pi/2,0,0,-np.pi,0])

viz.display(q0)

pin.framesForwardKinematics(human_model,human_data, q0)

for frame in human_model.frames.tolist():
    place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

