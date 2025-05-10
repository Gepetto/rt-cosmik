import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from viz_utils import place
import pandas as pd 
from src.rtcosmik.config_loader import settings


no_trial = "trial3"
task = "sit_to_stand"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/augmented_markers_filtred.csv"
mks_names = settings.marker_names

# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/3d_keypoints.csv"
# mks_names = [
#         "Nose", "LEye", "REye", "LEar", "REar", 
#         "LShoulder", "RShoulder", "LElbow", "RElbow", 
#         "LWrist", "RWrist", "LHip", "RHip", 
#         "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
#         "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
#     ]

# data = pd.read_csv(path_to_csv,skiprows=1).iloc[:,2:] #read mocap data skip first row cause header and 2 columns cause no frame
data = pd.read_csv(path_to_csv,skiprows=1)

assert len(data.columns) == 3 * len(mks_names), "The number of columns does not match the expected structure."

def csv_to_dict_of_dicts(df, headers):
    result = {}
    for i, header in enumerate(headers):
        # Each header corresponds to three consecutive columns (x, y, z)
        x_col = df.iloc[:, i*3]
        y_col = df.iloc[:, i*3 + 1]
        z_col = df.iloc[:, i*3 + 2]
        
        # Store x, y, z values as lists in the dictionary
        result[header] = {
            'x': x_col.tolist(),
            'y': y_col.tolist(),
            'z': z_col.tolist()
        }
    return result

dict_of_dicts_no_headers = csv_to_dict_of_dicts(data, mks_names)

viz = GepettoVisualizer()

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)


viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
viz.viewer.gui.addXYZaxis('world/torso', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/upperarm', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/lowerarm', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/pelvis', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/thigh', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/shank', [255, 0., 0, 1.], 0.01, 0.11)
viz.viewer.gui.addXYZaxis('world/foot', [255, 0., 0, 1.], 0.01, 0.11)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

num_frames = len(next(iter(dict_of_dicts_no_headers.values()))['x'])
mks_dict = []
# Loop through each frame to populate mks_dict
for i in range(num_frames):
    frame_data = {}
    for name, coordinates in dict_of_dicts_no_headers.items():
        # For each point, get the x, y, z of the current frame (i)
        x = coordinates['x'][i]
        y = coordinates['y'][i]
        z = coordinates['z'][i]
        
        # Store it as a numpy array for easy manipulation later
        frame_data[name] = np.array([x, y, z])
    
    # Append the frame data to mks_dict
    mks_dict.append(frame_data)


for name in mks_names:
    if name == "L_lwrist_study" or name == "r_lwrist_study" or name == "r_knee_study" or name == "L_knee_study" or name == "r_ankle_study" or name == "L_ankle_study" or name == "r_lelbow_study"or name == "L_lelbow_study" or name == 'r_5meta_study' or name == 'L_5meta_study':
        viz.viewer.gui.addSphere('world/'+name,0.01,[1,0,0,1])
    
    if name == "L.PSIS_study" or name == "r.PSIS_study":
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,1,0,1])
    else :
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,0,1,1])


for i in range(len(mks_dict)):
    for name in mks_names:
        # print(name)
        # print(mks_dict[i][name])
        sphere_name = f'world/{name}'
        place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(mks_dict[i][name].reshape(3,)).T))
    
    time.sleep(0.03)
    # input()


