import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd 
import numpy as np
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import parse_marker_csv,udp_csv_to_dataframe,read_mks_data,marker_data_to_dataframe
from viz_utils import place

no_trial = "Nicolas"
task = "static"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"
df_raw = pd.read_csv(path_to_csv)

marker_mocap_names = ['r.PSIS_study','L.PSIS_study','r.ASIS_study','L.ASIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

mks_names = marker_mocap_names
# df_wide = marker_data_to_dataframe(df_raw,mks_names)
df_wide = udp_csv_to_dataframe(path_to_csv, mks_names)
result_markers, start_sample_mks = read_mks_data(df_wide)


# === Initialiser le visualiseur Gepetto ===
viz = GepettoVisualizer()
try:
    viz.initViewer()
except ImportError as err:
    print("Install gepetto-viewer.")
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Start gepetto-viewer before running this script.")
    sys.exit(0)

viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.zeros((3,1))))

# === Ajouter les sphères ===
for name in mks_names:
    if name == "L_lwrist_study" or name == "r_lwrist_study" or name == "r_knee_study" or name == "L_knee_study" or name == "r_ankle_study" or name == "L_ankle_study" or name == "r_lelbow_study"or name == "L_lelbow_study" or name == 'r_5meta_study' or name == 'L_5meta_study':
        viz.viewer.gui.addSphere('world/'+name,0.01,[1,0,0,1])
    
    if name == "L.PSIS_study" or name == "r.PSIS_study":
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,1,0,1])
    else :
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,0,1,1])

# === Visualiser frame par frame ===
for i in range(0,len(result_markers)):
    for marker in result_markers[i].keys():
        place(viz, f'world/{marker}', pin.SE3(np.eye(3), np.matrix(result_markers[i][marker].reshape(3,)).T))
    
    # Avance image par image (appuie entrée)
    input(f"Frame {i+1}/{len(start_sample_mks)} - Press Enter")