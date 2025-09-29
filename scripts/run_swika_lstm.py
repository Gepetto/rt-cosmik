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
from src.rtcosmik.utils.read_write_utils import read_mks_data
import pandas as pd
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker
from src.rtcosmik.config_loader import settings
from src.rtcosmik.ik.ik import RT_SWIKA
from collections import deque
from src.rtcosmik.human_model.urdf_model import * 



mks_to_skip = ['TV8','TV12','SJN','STRN','LForearm','LUArm', 'RUArm','RHJC_study','LHJC_study',
               'LHand2','LHand1','LHL2','LHM5', 'RForearm','RHand2','RHand1','RHL2','RHM5']

no_trial = "Mohamed"
subject_height = 1.80
task = "bolting"
gender = 'male'
path_to_csv = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/augmented_markers.csv"
path_to_kpt = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/3d_keypoints_filtered.csv"

keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']

data_markers_lstm = pd.read_csv(path_to_csv) 
keypoints = pd.read_csv(path_to_kpt) 

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

if len(data_markers_lstm) != len(keypoints):
    raise ValueError("Row count mismatch between data_markers_lstm and keypoints")

data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

start_sample=0

result_markers, start_sample_dict = read_mks_data(data_markers_lstm, start_sample=start_sample) #check the function of read 

#load urdf
human = Robot('/home/msabbah/pinocchio-3x/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender=gender,subject_height=subject_height)
print(human_model.nq)
human_model= mks_registration(human_model,start_sample_dict, with_hand=False)
human_data = pin.Data(human_model)

################################################################################LOCK JOINTS
all_joint_ids = set(range(1, human_model.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if human_model.existJointName(jn):
        joint_ids_to_lock.append(human_model.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(human_model)
# Build reduced model
human_model, human_visual_model = pin.buildReducedModel(
    human_model, human_visual_model, joint_ids_to_lock, q0)

print(human_model.nq)
human_data = pin.Data(human_model)
###############################################################################################################

# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,start_sample_dict)
#model markers spheres 
add_marker(viz,result_markers[1].keys(),'_m', 0, 0,1)

q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)
viz.display(q)

dt = 40
N = 3
ik_code = settings.ik_code
cost_weights = settings.cost_weights
keys_to_track_list = ['Nose', 'Head', 'REye', 'LEye',
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study',
                        ]

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
    input()

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
    pin.forwardKinematics(human_model,human_data, q)
    pin.updateFramePlacements(human_model,human_data)
    for frame in human_model.frames.tolist():
        viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
        place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
    
    viz.display(q)
    q_list.append(q)


#save mks est (model markers)
df = pd.DataFrame(M_model_list)
csv_file = os.path.join(rt_cosmik_path,f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}//mks_model_swika.csv") 
df.to_csv(csv_file, index=False)

# #save angles
joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']
num_values = len(q_list[0])
if len(joint_angles_names) != num_values:
    raise ValueError(f"joint_angles_names has {len(joint_angles_names)} entries but q has {num_values} DOFs.")

df = pd.DataFrame(q_list, columns=joint_angles_names)
csv_file = os.path.join(rt_cosmik_path, f"output/{no_trial}/cosmik_2cams/{task}/q_cosmik_swika.csv")
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

