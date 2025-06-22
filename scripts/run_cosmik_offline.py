import os
import sys
# Add the src folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
script_directory = os.path.dirname(os.path.abspath(__file__))
rt_cosmik_path = os.path.dirname(script_directory)
from collections import deque
import numpy as np
import pandas as pd

from src.rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from src.rtcosmik.triangulation.triangulation import triangulate_offline
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv, load_transformation, transform_keypoints_list_cam0_to_mocap
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from src.rtcosmik.utils.read_write_utils import read_mks_data
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from src.rtcosmik.ik.ik import RT_IK
import gepetto as gep

# === Configuration ===
base_path = "/root/workspace/ros_ws/src/rt-cosmik"
no_trial = "Maxime"
task = "bolting"
augmenter_path = os.path.join(base_path, "src/rtcosmik/augmenter/augmentation_model")
transformation_file = f"{base_path}/output/{no_trial}/calib_mocap_2_cam0/soder.txt"

# === Marker headers ===
num_keypoints = 26
pose_markers = [
    "Nose", "LEye", "REye", "LEar", "REar", 
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]
pose_header = [f"{marker}_{axis}" for marker in pose_markers for axis in ['x', 'y', 'z']]

augmented_markers = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study','r_mknee_study',
    'r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study','r_calc_study','L_knee_study',
    'L_mknee_study','L_ankle_study','L_mankle_study','L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study','r_thigh1_study','r_thigh2_study',
    'r_thigh3_study','L_thigh1_study','L_thigh2_study','L_thigh3_study','r_sh1_study',
    'r_sh2_study','r_sh3_study','L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study',
    'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study',
    'L_melbow_study','L_lwrist_study','L_mwrist_study'
]
augmented_header = [f"{marker}_{axis}" for marker in augmented_markers for axis in ['x', 'y', 'z']]

# === Subject physical info for LSTM ===
subject_mass = 75.0
subject_height = 1.85

def run_ik_pipeline(augmented_csv_path, keypoints_csv_path, meshes_folder_path, output_q_csv_path, trial_name, task_name):
    start_sample =0 
    mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis',
               'LHand','LHL2','LHM5', 'RForearm','RHand','RHL2','RHM5', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']

    data_markers_lstm = pd.read_csv(augmented_csv_path)
    keypoints = pd.read_csv(keypoints_csv_path)

    # === Ajouter markers faciaux (caméra) aux marqueurs LSTM
    keys_to_add = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye']
    columns_to_add = [col for col in keypoints.columns if any(k + '_' in col for k in keys_to_add)]
    if len(data_markers_lstm) != len(keypoints):
        raise ValueError("Row count mismatch between LSTM markers and keypoints")

    data_markers_lstm = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

    result_markers, start_sample_dict = read_mks_data(data_markers_lstm, start_sample=0)
    start_sample_dict = result_markers[0]

    #load urdf
    human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf',rt_cosmik_path,isFext=True) 
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    #scale the model to data
    human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender='male',subject_height=1.85)
    print(human_model.nq)

    human_model= mks_registration(human_model,start_sample_dict, with_hand=False)

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
    # seg_frames = construct_segments_frames(start_sample_dict)
    # add_frames(viz,seg_frames,"meas", 0.008, 0.08)

    #model markers spheres 
    add_marker(viz,result_markers[1].keys(),'_m', 1, 0,0)

    # IK init
    q = pin.neutral(human_model)
    human_data = pin.Data(human_model)
    dt = 1 / 40.0

    keys_to_track = ['Nose', 'Head', 'REar', 'LEar', 'REye', 'LEye',
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

    ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track, dt)
    q = ik_class.solve_ik_sample_casadi()
    viz.display(q)
    ik_class._q0 = q

    q_list, rmse_per_marker, M_model_list = [], {}, []

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

    #save mks est
    df = pd.DataFrame(M_model_list)
    csv_file = os.path.join(rt_cosmik_path,f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_model_cosmik_ipopt.csv") 
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


def main():
    num_keypoints = 26
    # === Paths ===
    config_path = os.path.join(base_path, "config/cam_params")
    file_paths = [
        os.path.join(base_path, f"output/{no_trial}/{task}/keypoints_0.csv"),
        os.path.join(base_path, f"output/{no_trial}/{task}/keypoints_2.csv")
    ]
    filtered_kpt_path = os.path.join(base_path, f"output/{no_trial}/{task}/3d_keypoints_filtred.csv")
    augmented_output_path = os.path.join(base_path, f"output/{no_trial}/{task}/augmented_markers.csv")

    # === Load MoCap transformation ===
    R_trans, d_trans, s_trans, rms_error = load_transformation(transformation_file)

    # === Load 2D keypoints from cameras ===
    camera_data = [read_mmpose_file(fp) for fp in file_paths]
    uvs = [
        np.array([[line[2*i], line[2*i + 1]] for line in data for i in range(num_keypoints)])
        .reshape(-1, num_keypoints, 2)
        for data in camera_data
    ]

    # === Load camera calibration ===
    mtxs, dists, projections, rotations, translations = load_camera_parameters(config_path)
    world_R1_cam, world_T1_cam = load_world_transformation(config_path)

    # === Triangulate 3D keypoints ===
    keypoints_cam0 = triangulate_offline(uvs, mtxs, dists, projections, world_R1_cam, world_T1_cam)

    # === Transform to MoCap frame ===
    keypoints_mocap = transform_keypoints_list_cam0_to_mocap(keypoints_cam0, R_trans, d_trans)

    # === Filter 3D keypoints ===
    keypoints_filtered = butterworth_filter(
        data=keypoints_mocap,
        cutoff_frequency=10.0,
        order=5,
        sampling_frequency=40
    )
    save_to_csv(keypoints_filtered, filtered_kpt_path, header=pose_header)

    # === Load LSTM augmenter model ===
    lstm_model = loadModel(
        augmenterDir=augmenter_path,
        augmenterModelName="LSTM",
        augmenter_model='v0.3'
    )

    # === Apply augmentation model (sliding window over 30 frames) ===
    buffer = deque(maxlen=30)
    augmented_output = []
    first_frame = True
    num_keypoints = keypoints_filtered.shape[1] // 3

    for i in range(len(keypoints_filtered)):
        frame_data = keypoints_filtered[i].reshape(num_keypoints, 3)

        if first_frame:
            for _ in range(30):
                buffer.append(frame_data)
            first_frame = False
        else:
            buffer.append(frame_data)

        if len(buffer) == 30:
            buffer_array = np.array(buffer)
            augmented_frame = augmentTRC(
                buffer_array,
                subject_mass=subject_mass,
                subject_height=subject_height,
                models=lstm_model,
                augmenterDir=augmenter_path,
                augmenter_model='v0.3'
            )
            augmented_output.append(augmented_frame)

    # === Save augmented anatomical markers ===
    augmented_array = np.vstack(augmented_output)
    save_to_csv(augmented_array, augmented_output_path, header=augmented_header)

    # === run inverse kinematics ===
    run_ik_pipeline(
    augmented_csv_path=augmented_output_path,
    keypoints_csv_path=filtered_kpt_path,
    meshes_folder_path=os.path.join(base_path, "meshes"),
    output_q_csv_path=os.path.join(base_path, f"output/{no_trial}/{task}/q_cosmik_ipopt.csv"),
    trial_name=no_trial,
    task_name=task
)

if __name__ == "__main__":
    main()
