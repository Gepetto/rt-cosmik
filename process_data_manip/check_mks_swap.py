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
from utils.read_write_utils import read_mks_data
from utils.model_w_mocap_utils import build_model_challenge, get_segments_lstm_mks_dict_challenge, get_segments_mks_dict, construct_segments_frames_challenge
import cv2
from utils.linear_algebra_utils import rotation_matrix_angle, rodrigues_angle


cleaned_data = []
prev_rot = {}
threshold_angle = 50 #degrees

fichier_csv_mks = 'mks_mocap/mks_mocap_test_2.csv'
data = pd.read_csv(fichier_csv_mks)
start_sample = 0
result_markers, mks_dict = read_mks_data(data, start_sample=start_sample)

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

#frames from mks
seg_frames = construct_segments_frames_challenge(result_markers[start_sample])
for seg_name, mks in seg_frames.items():
    frame_name = f'world/{seg_name+"_meas"}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.008, 0.08)


viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

#mks
for marker in result_markers[1].keys():
    if marker == "L_lwrist_study" or marker == "r_lwrist_study" or marker == "r_knee_study" or marker == "L_knee_study" or marker == "r_ankle_study" or marker == "L_ankle_study" or marker == "r_lelbow_study"or marker == "L_lelbow_study" or marker == 'r_5meta_study' or marker == 'L_5meta_study':
        viz.viewer.gui.addSphere('world/'+marker,0.01,[1,0,0,1])
    
    if marker == "L.PSIS_study" or marker == "r.PSIS_study":
        viz.viewer.gui.addSphere('world/'+marker,0.01,[0,1,0,1])
    else :
        viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])


#display mks and frames
for ii in range(start_sample,len(result_markers)):

    mks_dict = result_markers[ii]
    row_cleaned = mks_dict.copy()
    
    for marker in mks_dict.keys():
        viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])
        M = pin.SE3(pin.SE3(np.eye(3), np.matrix([mks_dict[marker][0],mks_dict[marker][1],mks_dict[marker][2]]).T))
        place(viz,'world/'+marker,M)
    
    #Display frames from measurements and dectect any discontinuities using angles
    seg_frames = construct_segments_frames_challenge(row_cleaned)
    for seg_name, rot in seg_frames.items():
        frame_name = f'world/{seg_name+"_meas"}'
        frame_se3 = pin.SE3(rot[:3,:3], np.matrix([rot[0,3],rot[1,3],rot[2,3]]).T)
        place(viz, frame_name, frame_se3)
        
        #check if there is discontinuities if yes re-write the data by swapping data of markers 
        if seg_name in prev_rot: 
            angle = rotation_matrix_angle(rot[:3,:3], prev_rot[seg_name])
            
            if angle > threshold_angle:
                print(angle)
                print(ii)
                marker1, marker2 = get_segments_mks_dict()[seg_name]  
                row_cleaned[marker1], row_cleaned[marker2] = row_cleaned[marker2], row_cleaned[marker1]
                print(" rot mat: ", rot[:3,:3])
                print(get_segments_mks_dict()[seg_name])

            #get the new rot with the cleaned data     
            seg_frames = construct_segments_frames_challenge(row_cleaned)
            rot = seg_frames.get(seg_name)
            # Update the previous Rodrigues vector
            prev_rot[seg_name] = rot[:3,:3].copy()
        else : 
            prev_rot[seg_name] = rot[:3,:3].copy()
            # print(marker)
    
    flattened_row = {}
    for marker, pos in row_cleaned.items():
        flattened_row[f'{marker}_x'] = pos[0]
        flattened_row[f'{marker}_y'] = pos[1]
        flattened_row[f'{marker}_z'] = pos[2]
    
    cleaned_data.append(flattened_row)


    # input()
    # time.sleep(0.03)
# df = pd.DataFrame(cleaned_data)
# df.to_csv('mks_mocap_test_2_corrected', index=False)