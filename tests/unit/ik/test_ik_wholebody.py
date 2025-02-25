import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
meshes_folder_path = '/root/workspace/ros_ws/src/rt-cosmik/meshes/old_meshes'
rt_cosmik_path = os.path.dirname(script_directory)
from human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from utils.read_write_utils import read_mks_data
import pandas as pd
from human_model.pin_model import * 
from human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from ik.ik import RT_IK

#read mks data
start_sample = 0
data_markers = pd.read_csv('/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/augmented_data.csv') 

# read mks data
result_markers, start_sample_dict = read_mks_data(data_markers, start_sample=start_sample) #check the function of read 
#build/calibrate human model
human_model, human_geom_model, visuals_dict = build_model(start_sample_dict,meshes_folder_path)

# VISUALIZATION

viz = gv_init(human_model,human_geom_model.copy(),human_geom_model,start_sample_dict.keys())
#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)

#model markers spheres 
add_marker(viz,result_markers[1].keys(), 1, 0,0)
#model frames
seg_names_mks = get_segments_mks_dict()
add_frames(viz,seg_names_mks,"model", 0.012, 0.05)



### IK init 
q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)

dt = 1/40 #dt for qp
keys_to_track_list = [  'C7_study', 
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


q_list = []
M_model_list = []

for ii in range(start_sample,len(result_markers)): 

    mks_dict = result_markers[ii]
    ik_class._dict_m= mks_dict
    q = ik_class.solve_ik_sample_quadprog() 

    pin.forwardKinematics(human_model, human_data, q)
    pin.updateFramePlacements(human_model, human_data)
    
    M_model_frame = {}

    for marker in result_markers[ii].keys():
        if marker == "LHJC_study" or marker == "RHJC_study":
            continue  #skip

        M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
        M_model = human_data.oMf[human_model.getFrameId(marker)]
        # M_model = pin.SE3(Rquat(1, 0, 0, 0), np.matrix([M_model.translation[0],M_model.translation[1],M_model.translation[2]]).T)

        # Add marker_model position to the frame's data
        M_model_frame[f"{marker}_x"] = M_model.translation[0]
        M_model_frame[f"{marker}_y"] = M_model.translation[1]
        M_model_frame[f"{marker}_z"] = M_model.translation[2]
        
        place(viz,'world/'+marker,M)
        place(viz,'world/'+marker+"_m",M_model)

    M_model_list.append(M_model_frame)

    #save mks est
    # df = pd.DataFrame(M_model_list)
    # csv_file = os.path.join(rt_cosmik_path,'process_data_manip/mks_cosmik_model_test_2_modele_mocap.csv') 
    # df.to_csv(csv_file, index=False)
    
    #Display frames from human_model
    for seg_name, mks in seg_names_mks.items():
        
        frame_name = f'world/{seg_name+"_model"}'
        frame_se3= human_data.oMf[human_model.getFrameId(seg_name)]
        place(viz, frame_name, frame_se3)
    
    #Display frames from measurements
    seg_frames = construct_segments_frames(mks_dict)
    for seg_name, M in seg_frames.items():
        
        frame_name = f'world/{seg_name+"_meas"}'
        frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
        place(viz, frame_name, frame_se3)

    #display q
    viz.display(q)
    input("Press Enter to continue...")
    ik_class._q0 = q 

    q_list.append(q)
    
    

#save angles
# num_values = len(q_list[0])  
# headers = [f"q{i}" for i in range(num_values)]
# df = pd.DataFrame(q_list, columns=headers)

# csv_file = os.path.join(rt_cosmik_path,'process_data_manip/q_cosmik_qp_modele_mocap.csv') 
# df.to_csv(csv_file, index=False)
