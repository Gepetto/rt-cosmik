import sys
import os
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from read_csv_file import csv_to_dict_of_dicts
from viz_utils import place
import pandas as pd 


# mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','C7_study','r_knee_study','r_mknee_study','L_knee_study','L_mknee_study','r_ankle_study','r_mankle_study','L_ankle_study','L_mankle_study',
#                   'r_toe_study','r_5meta_study','L_toe_study','L_5meta_study','r_calc_study','L_calc_study','r_shoulder_study','L_shoulder_study',
#                   'r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study','L_thigh2_study','L_thigh3_study',
#                   'r_sh1_study','r_sh2_study','r_sh3_study','L_sh1_study','L_sh2_study','L_sh3_study',
#                   'r_lelbow_study','r_melbow_study','L_lelbow_study','L_melbow_study',
#                   'r_lwrist_study','r_mwrist_study','L_lwrist_study','L_mwrist_study']  #mocap data

mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

# mks_names = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","Rknee","LAnkle","RAnkle","Head","Neck","Hip","LBigToe","RBigToe","LSmallToe", "RSmallToe", "LHeel","RHeel"]


fichier_csv_mks = '/root/workspace/ros_ws/src/rt-cosmik/output/lstm.csv'
# data = pd.read_csv(fichier_csv_mks, header=None)  #read all data (pas de ligne en trop)

# fichier_csv_mks = 'mks_mocap_test_2.csv'
data = pd.read_csv(fichier_csv_mks,skiprows=1) #.iloc[:,1:] #read mocap data skip first row cause header and first column cause time
assert len(data.columns) == 3 * len(mks_names), "The number of columns does not match the expected structure."

# fichier_csv_mocap_mks = "data/mks_coordinates_3D.trc"
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
    
    # time.sleep(0.03)
    input()


