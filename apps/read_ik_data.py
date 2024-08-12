from utils.model_utils import Robot
import pandas as pd
import sys
from pinocchio.visualize import GepettoVisualizer

# Loading human urdf
human = Robot('models/human_urdf/urdf/human.urdf','models') 
human_model = human.model
human_visual_model = human.visual_model
human_collision_model = human.collision_model

q=pd.read_csv('output/q.csv').to_numpy()[:,-5:]
print(q)

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

for ii in range(q.shape[0]):
    q_ii = q[ii,:]
    viz.display(q_ii)
    input()