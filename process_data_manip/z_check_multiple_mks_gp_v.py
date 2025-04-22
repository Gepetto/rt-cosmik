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


mocap_mks_names = ['r.ASIS_study_','L.ASIS_study_','r.PSIS_study_','L.PSIS_study_','C7_study_','r_knee_study_','r_mknee_study_','L_knee_study_','L_mknee_study_','r_ankle_study_','r_mankle_study_','L_ankle_study_','L_mankle_study_',
                  'r_toe_study_','r_5meta_study_','L_toe_study_','L_5meta_study_','r_calc_study_','L_calc_study_','r_shoulder_study_','L_shoulder_study_',
                  'r_thigh1_study_','r_thigh2_study_','r_thigh3_study_','L_thigh1_study_','L_thigh2_study_','L_thigh3_study_',
                  'r_sh1_study_','r_sh2_study_','r_sh3_study_','L_sh1_study_','L_sh2_study_','L_sh3_study_',
                  'r_lelbow_study_','r_melbow_study_','L_lelbow_study_','L_melbow_study_',
                  'r_mwrist_study_','r_lwrist_study_','L_mwrist_study_','L_lwrist_study_']  #mocap data

lstm_mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

# lstm_mks_names = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","Rknee","LAnkle","RAnkle","Head","Neck","Hip","LBigToe","RBigToe","LSmallToe", "RSmallToe", "LHeel","RHeel"]

fichier_csv_lstm_mks = 'mks_lstm/augmented_markers_positions_by_rows_with_header.csv' 
data = pd.read_csv(fichier_csv_lstm_mks,skiprows=1).iloc[:,1:]
# fichier_csv_lstm_mks = 'mks_lstm/augmented_mks_offline.csv' #lstm data not filtred
# data = pd.read_csv(fichier_csv_lstm_mks, header=None) #read all data (pas de ligne en trop)

fichier_csv_mocap_mks = 'mks_mocap/mks_mocap_test_2.csv'

# fichier_csv_lstm_mks = '/home/kahina/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3/example/python/output0/keypoints_3d_positions_3.csv'

data_mocap = pd.read_csv(fichier_csv_mocap_mks,skiprows=1).iloc[:,1:]
data_mocap = data_mocap[:]
assert len(data.columns) == 3 * len(lstm_mks_names), "The number of columns does not match the expected structure."

# fichier_csv_mocap_mks = "data/mks_coordinates_3D.trc"
dict_of_dicts_no_headers = csv_to_dict_of_dicts(data, lstm_mks_names)
dict_of_dicts_no_headers_mocap = csv_to_dict_of_dicts(data_mocap, mocap_mks_names)


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
lstm_mks_dict = []
# Loop through each frame to populate lstm_mks_dict
for i in range(num_frames):
    frame_data = {}
    for name, coordinates in dict_of_dicts_no_headers.items():
        # For each point, get the x, y, z of the current frame (i)
        x = coordinates['x'][i]
        y = coordinates['y'][i]
        z = coordinates['z'][i]
        
        # Store it as a numpy array for easy manipulation later
        frame_data[name] = np.array([x, y, z])
    
    # Append the frame data to lstm_mks_dict
    lstm_mks_dict.append(frame_data)



num_frames_ = len(next(iter(dict_of_dicts_no_headers_mocap.values()))['x'])
mocap_mks_dict = []
# Loop through each frame to populate lstm_mks_dict
for i in range(num_frames_):
    frame_data_ = {}
    for name, coordinates in dict_of_dicts_no_headers_mocap.items():
        # For each point, get the x, y, z of the current frame (i)
        x = coordinates['x'][i]
        y = coordinates['y'][i]
        z = coordinates['z'][i]
        
        # Store it as a numpy array for easy manipulation later
        frame_data_[name] = np.array([x, y, z])
    
    # Append the frame data to lstm_mks_dict
    mocap_mks_dict.append(frame_data_)

for name in lstm_mks_names:
    sphere_n = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_n, 0.015, [0, 0., 255, 1.])

for name in mocap_mks_names:
    sphere_name = f'world/{name}'
    viz.viewer.gui.addSphere(sphere_name, 0.015, [0, 255, 255, 1.])


for i in range(0,5):
    for name in mocap_mks_names:
        sphere_name = f'world/{name}'
        place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(mocap_mks_dict[i][name].reshape(3,)).T))

    for n in lstm_mks_names:
        sphere_n= f'world/{n}'
        place(viz, sphere_n, pin.SE3(np.eye(3), np.matrix(lstm_mks_dict[i][n].reshape(3,)).T))
    
    time.sleep(0.03)


