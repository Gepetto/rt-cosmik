import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/'
rt_cosmik_path = os.path.dirname(script_directory)
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe
import pandas as pd
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.pin_model import build_model
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from src.rtcosmik.ik.ik import RT_IK,RT_SWIKA



no_trial = "trial3"
task = "polissage_robot"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/augmented_markers_filtred.csv"
path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/3d_keypoints.csv"

keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']

data_markers_lstm = pd.read_csv(path_to_csv) 
keypoints = pd.read_csv(path_to_kpt) 

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

if len(data_markers_lstm) != len(keypoints):
    raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

start_sample=0

result_markers, start_sample_dict = read_mks_data(data_markers_lstm, start_sample=start_sample) #check the function of read 
start_sample_dict = result_markers[start_sample]


human_model, human_geom_model, visuals_dict = build_model(start_sample_dict, meshes_folder_path)


# VISUALIZATION

viz = GepettoVisualizer(human_model,human_geom_model.copy(),human_geom_model)
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

import gepetto as gep
viz.viewer.gui.setBackgroundColor1("python-pinocchio", gep.color.Color.white)
viz.viewer.gui.setBackgroundColor2("python-pinocchio", gep.color.Color.white)
viz.viewer.gui.addLight("light", "python-pinocchio", 360, gep.color.Color.white)

#markers spheres (model and measured)
for marker in result_markers[1].keys():
    if marker == "L_lwrist_study" or marker == "r_lwrist_study":
        viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
        viz.viewer.gui.addSphere('world/'+marker+"_m",0.01,[0,1,0,1])
    else :
        viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
        viz.viewer.gui.addSphere('world/'+marker+"_m",0.01,[0,1,0,1])

#model frames
seg_names_mks = get_segments_mks_dict(result_markers[start_sample])

for seg_name, mks in seg_names_mks.items():
    frame_name = f'world/{seg_name}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.012, 0.05)

#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
for seg_name, mks in seg_frames.items():
    frame_name = f'world/{seg_name+"_meas"}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.008, 0.08)


### IK init 
q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)

dt = 1/40 #dt for qp
keys_to_track_list = [  'Head', 'Nose', 'REar', 'LEar', 'REye', 'LEye',
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'L_sh1_study', 'L_sh2_study', 'L_sh3_study'
                        ]

### IK calculations
ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt)
q = ik_class.solve_ik_sample_casadi() #warm start with ipopt for qp 
viz.display(q)
ik_class._q0=q


print(q)
input()

rmse_per_marker = {}
q_list = []
M_model_list = []

for ii in range(start_sample,len(result_markers)): 
    print(ii)

    lstm_dict = result_markers[ii]
    # ik_class._dict_m= lstm_dict
    ik_class = RT_IK(human_model, lstm_dict, q, keys_to_track_list, dt)

    q = ik_class.solve_ik_sample_casadi() 
    # print("q", len(q))
    # print(q)
    pin.forwardKinematics(human_model, human_data, q)
    pin.updateFramePlacements(human_model, human_data)
    
    M_model_frame = {}

    for marker in result_markers[ii].keys():
        if marker == "LHJC_study" or marker == "RHJC_study":
        # if ((marker != "C7_study") & (marker != "r.ASIS_study") & (marker != "L.ASIS_study") & (marker != "r.PSIS_study") & (marker != "L.PSIS_study")
        # &  (marker != "L_knee_study")& (marker != "L_mknee_study")):
            continue  #skip
        
        pos_gt = np.array(result_markers[ii][marker])
        M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
        M_model = human_data.oMf[human_model.getFrameId(marker)]
        # M_model = pin.SE3(Rquat(1, 0, 0, 0), np.matrix([M_model.translation[0],M_model.translation[1],M_model.translation[2]]).T)
        pos_model = np.array(M_model.translation).flatten()

        # Add marker_model position to the frame's data
        M_model_frame[f"{marker}_x"] = M_model.translation[0]
        M_model_frame[f"{marker}_y"] = M_model.translation[1]
        M_model_frame[f"{marker}_z"] = M_model.translation[2]
        
        place(viz,'world/'+marker,M)
        place(viz,'world/'+marker+"_m",M_model)

        # RMSE calculation
        sq_error = np.sum((pos_gt - pos_model) ** 2)

        if marker not in rmse_per_marker:
            rmse_per_marker[marker] = []
        rmse_per_marker[marker].append(sq_error)

    M_model_list.append(M_model_frame)

    # #save mks est
    # df = pd.DataFrame(M_model_list)
    # csv_file = os.path.join(rt_cosmik_path,'process_data_manip/mks_cosmik_model_test_2_modele_mocap.csv') 
    # df.to_csv(csv_file, index=False)
    

    #Display frames from human_model
    for seg_name, mks in seg_names_mks.items():
        
        frame_name = f'world/{seg_name}'
        frame_se3= human_data.oMf[human_model.getFrameId(seg_name)]
        place(viz, frame_name, frame_se3)
    
    #Display frames from measurements
    seg_frames = construct_segments_frames(lstm_dict)
    for seg_name, M in seg_frames.items():
        
        frame_name = f'world/{seg_name+"_meas"}'
        frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
        place(viz, frame_name, frame_se3)

    #display q
    viz.display(q)
    # input("Press Enter to continue...")
    ik_class._q0 = q 

    q_list.append(q)
  
#save angles

joint_angles_names = settings.joint_angles_names
num_values = len(q_list[0])
if len(joint_angles_names) != num_values:
    raise ValueError(f"joint_angles_names has {len(joint_angles_names)} entries but q has {num_values} DOFs.")

df = pd.DataFrame(q_list, columns=joint_angles_names)
csv_file = os.path.join(rt_cosmik_path, f"output/{no_trial}/{task}/q_cosmik_ipopt.csv")
df.to_csv(csv_file, index=False)


rmse_global = 0
nb_mks =0 
# Final RMSE output
print("\nPer-marker RMSE (in meters):")
for marker, sq_errors in rmse_per_marker.items():
    nb_mks +=1
    rmse = np.sqrt(np.mean(sq_errors))
    print(f"{marker}: {rmse:.4f} m")
    rmse_global +=rmse

rmse_global = rmse_global/nb_mks
print(f" Global RMSE across all markers and frames: {rmse_global:.4f} m")

