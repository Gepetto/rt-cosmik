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
from collections import deque



mks_to_skip = ['TV8','TV12','SJN','STRN','LForearm','LUArm', 'RUArm','RHJC_study','LHJC_study',
               'LHand2','LHand1','LHL2','LHM5', 'RForearm','RHand2','RHand1','RHL2','RHM5']

no_trial = "trial_2"
task = "trial_upper3"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/markers.csv"
path_to_kpt = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/keypoints.csv"

keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']

data_markers_lstm = pd.read_csv(path_to_csv) 
keypoints = pd.read_csv(path_to_kpt) 

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

if len(data_markers_lstm) != len(keypoints):
    raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

start_sample=0

result_markers, start_sample_mks = read_mks_data(data_markers_lstm, start_sample=start_sample) #check the function of read 

#build and scale the model in sample 0 
human_model, human_geom_model, visuals_dict = build_model(start_sample_mks,meshes_folder_path)

# VISUALIZATION
viz = gv_init(human_model,human_geom_model.copy(),human_geom_model,start_sample_mks.keys())
#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)
#model markers spheres 
add_marker(viz,result_markers[1].keys(), 0, 1,0)
#model frames
seg_names_mks = get_segments_mks_dict(result_markers[start_sample])
add_frames(viz,seg_names_mks,"model", 0.012, 0.05)

q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)
viz.display(q)

dt = settings.dt
N = settings.N
ik_code = settings.ik_code
cost_weights = settings.cost_weights
keys_to_track_list = settings.keys_to_track_list

### IK calculations

### IK init 
ik_class = RT_SWIKA(human_model, keys_to_track_list, N, code = ik_code)
x_array = np.zeros((human_model.nq+human_model.nv, N))
x_array[6,:]=1
u_array = np.zeros((human_model.nv, N))
deque_lstm_dict = deque(maxlen=N)


rmse_per_marker = {}
q_list=[]
M_model_list = []
deque_lstm_dict = deque(maxlen=N)

for ii in range(len(result_markers)):
    mks_dict = result_markers[ii]
    if ii ==0:
        for i in range(N):
            deque_lstm_dict.append(mks_dict)
    else:
        deque_lstm_dict.append(mks_dict)

    array_data = np.array([
        np.hstack([d[marker] for marker in keys_to_track_list])
        for d in deque_lstm_dict]).T
    # print(x_array)
    x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:, -1], cost_weights, dt)

    q = pin.neutral(human_model)
    q[:] = np.array(x_array[:human_model.nq, -1]).flatten()
    pin.forwardKinematics(human_model, human_data, q)
    pin.updateFramePlacements(human_model, human_data)

    M_model_frame = {}

    for marker in result_markers[ii].keys():
        if marker in mks_to_skip: 
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

    #Display frames from human_model
    for seg_name, mks in seg_names_mks.items():
        
        frame_name = f'world/{seg_name+"_model"}'
        frame_se3= human_data.oMf[human_model.getFrameId(seg_name)]
        place(viz, frame_name, frame_se3)
    
    
    viz.display(q)
    q_list.append(q)
    # input()


    #save mks est
# df = pd.DataFrame(M_model_list)
# csv_file = os.path.join(rt_cosmik_path,f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_model.csv") 
# df.to_csv(csv_file, index=False)

# #save angles
joint_angles_names = settings.joint_angles_names
num_values = len(q_list[0])
if len(joint_angles_names) != num_values:
    raise ValueError(f"joint_angles_names has {len(joint_angles_names)} entries but q has {num_values} DOFs.")

df = pd.DataFrame(q_list, columns=joint_angles_names)
csv_file = os.path.join(rt_cosmik_path, f"output/{no_trial}/{task}/q_cosmik_offline.csv")
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

