import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data,udp_csv_to_dataframe,marker_data_to_dataframe
import pandas as pd
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length
from src.rtcosmik.ik.ik import RT_IK


mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
               'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']
#read mks data
no_trial = "Gabriel"
task = "static"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"

start_sample=0
mks_names = ['r.PSIS_study','L.PSIS_study','r.ASIS_study','L.ASIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']
# df_raw = pd.read_8data_to_dataframe(df_raw, mks_names) #marker data are string 
mks_data = udp_csv_to_dataframe(path_to_csv, mks_names) #float
result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample) #check the function of read 

#load urdf
human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender='male',subject_height=1.85)
print(human_model.nq)

human_model= mks_registration(human_model,start_sample_dict, with_hand=True)

human_data = pin.Data(human_model)

# VISUALIZATION
viz = gv_init(human_model,human_collision_model,human_visual_model,start_sample_dict)
pin.forwardKinematics(human_model,human_data, pin.neutral(human_model))
pin.updateFramePlacements(human_model,human_data)

# display urdf frames
# for frame in human_model.frames.tolist():
#     viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
#     place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])

q =pin.neutral(human_model)

viz.display(q)
input("model scaled, you can launch ik")

#measured frames
seg_frames = construct_segments_frames(result_markers[start_sample])
add_frames(viz,seg_frames,"meas", 0.008, 0.08)

#model markers spheres 
add_marker(viz,result_markers[1].keys(),'_m', 1, 0,0)
#model frames
for joint_id in range(1, human_model.njoints):  # Skip 0 (universe)
    frame_name = f'world/{human_model.names[joint_id]+"_model"}'
    viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.012, 0.05)



### IK init 
q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)

dt = 1/40 #dt for qp
#track only real markers (without technical markers)
keys_to_track_list = [
        'BHD','RHD','LHD','FHD',
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
        'L_knee_study', 'L_mknee_study'
    ]

### IK calculations
ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt)
q = ik_class.solve_ik_sample_casadi() #warm start with ipopt for qp 
viz.display(q)
ik_class._q0=q
input('first sample')

rmse_per_marker = {}
q_list = []
M_model_list = []

for ii in range(start_sample,len(result_markers)): 

    mks_dict = result_markers[ii]
    ik_class._dict_m= mks_dict
    q = ik_class.solve_ik_sample_casadi() 

    pin.forwardKinematics(human_model, human_data, q)
    pin.updateFramePlacements(human_model, human_data)
    
    M_model_frame = {}

    for marker in result_markers[ii].keys():
        # print(marker)
        if marker in mks_to_skip: 
            continue  #skip
        pos_gt = np.array(result_markers[ii][marker])

        M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
        M_model = human_data.oMf[human_model.getFrameId(marker)]
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
    
    # Display frames from measurements
    # seg_frames = construct_segments_frames(mks_dict)
    # for seg_name, M in seg_frames.items():
        
    #     frame_name = f'world/{seg_name+"_meas"}'
    #     frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
    #     place(viz, frame_name, frame_se3)
    
    # #  Display frames from human_model
    # for joint_id in range(1, human_model.njoints):  # Skip 0 (universe)
    #     frame_name = f'world/{human_model.names[joint_id]+"_model"}'
    #     frame_se3= human_data.oMf[human_model.getFrameId(human_model.names[joint_id])]
    #     place(viz, frame_name, frame_se3)

    #display q
    viz.display(q)
    # input("Press Enter to continue...")
    ik_class._q0 = q 

    q_list.append(q)

# #save mks est
df = pd.DataFrame(M_model_list)
csv_file = os.path.join(rt_cosmik_path,f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_model_ipopt.csv") 
df.to_csv(csv_file, index=False)

#save angles
joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']
num_values = len(q_list[0])
if len(joint_angles_names) != num_values:
    raise ValueError(f"joint_angles_names has {len(joint_angles_names)} entries but q has {num_values} DOFs.")

df = pd.DataFrame(q_list, columns=joint_angles_names)
csv_file = os.path.join(rt_cosmik_path, f"output/{no_trial}/{task}/q_mocap_ipopt.csv")
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

