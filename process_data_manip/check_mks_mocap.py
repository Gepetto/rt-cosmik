import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd 
import numpy as np
import pinocchio as pin 
import time
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import parse_marker_csv
from src.rtcosmik.config_loader import settings
from  src.rtcosmik.utils.read_write_utils  import read_mks_data

# no_trial = "Mathis"
# task = "bolting"
# path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/{task}_trajectories.csv"
# df = pd.read_csv(path_to_csv)
# df.columns = [col.replace(f"{no_trial}:", "") for col in df.columns]
# frames = df["Frame"] if "Frame" in df.columns else range(len(df))
# mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))
path_to_csv_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/4279/mocap/robot_welding/mocap_downsampled_to_40hz.csv"

mks_dict, start_sample_dict = read_mks_data(df, start_sample=0) #convert to m if needed 

# mks_names = settings.marker_mocap_names
# mks_dict = parse_marker_csv(path_to_csv, mks_names)

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
for label in ['torso', 'upperarm', 'lowerarm', 'pelvis', 'thigh', 'shank', 'foot']:
    viz.viewer.gui.addXYZaxis(f'world/{label}', [255, 0., 0, 1.], 0.01, 0.11)

def place(viewer, name, pose):
    viewer.viewer.gui.applyConfiguration(name, list(pose.translation) + [0, 0, 0, 1])
    viewer.viewer.gui.refresh()

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
for i, frame in enumerate(mks_dict):
    for name in mks_names:
        pos = frame[name].reshape(3,)
        place(viz, f'world/{name}', pin.SE3(np.eye(3), pos.reshape(3,1)))
    
    # Avance image par image (appuie entrée)
    # input(f"Frame {i+1}/{len(mks_dict)} - Press Enter")