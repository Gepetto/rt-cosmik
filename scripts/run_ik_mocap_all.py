import os
import sys
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from src.rtcosmik.human_model.urdf_model import *
from src.rtcosmik.utils.read_write_utils import read_mks_data, udp_csv_to_dataframe
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.human_model.model_utils import get_segment_length
from src.rtcosmik.ik.ik import RT_IK


def run_ik(task, no_trial="Anais", subject_mass=72.0, subject_height=1.80, gender='male', start_sample=0):
    rt_cosmik_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mks_data.csv"

    mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
                   'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']

    mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']
    
    mks_data = udp_csv_to_dataframe(path_to_csv, mks_names)
    result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample)

    # Load URDF
    human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf', rt_cosmik_path, isFext=True)
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    human_model = scale_human_model(human_model, start_sample_dict, with_hand=True, gender=gender, subject_height=subject_height)
    human_model = mks_registration(human_model, start_sample_dict, with_hand=True)
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
    viz = gv_init(human_model, human_collision_model, human_visual_model, start_sample_dict)
    pin.forwardKinematics(human_model, human_data, pin.neutral(human_model))
    pin.updateFramePlacements(human_model, human_data)

    for frame in human_model.frames.tolist():
        viz.viewer.gui.addXYZaxis('world/' + frame.name, [1, 0, 0, 1], 0.01, 0.1)
        place(viz, 'world/' + frame.name, human_data.oMf[human_model.getFrameId(frame.name)])

    q = pin.neutral(human_model)
    human_data = pin.Data(human_model)
    viz.display(q)
    # input("Model scaled, you can launch IK")
    #measured frames
    seg_frames = construct_segments_frames(result_markers[start_sample])
    add_frames(viz, seg_frames, "meas", 0.008, 0.08)
    #model markers spheres 
    add_marker(viz, result_markers[1].keys(), '_m', 1, 0, 0)
    #model frames
    for joint_id in range(1, human_model.njoints):
        frame_name = f'world/{human_model.names[joint_id] + "_model"}'
        viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.012, 0.05)

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

    ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt=1/40)
    q = ik_class.solve_ik_sample_casadi()
    viz.display(q)
    ik_class._q0 = q
    # input("First sample")

    rmse_per_marker = {}
    q_list = []
    M_model_list = []

    for ii in range(start_sample, len(result_markers)):
        mks_dict = result_markers[ii]
        ik_class._dict_m = mks_dict
        q = ik_class.solve_ik_sample_casadi()
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        M_model_frame = {}

        for marker in result_markers[ii].keys():
            if marker in mks_to_skip:
                continue
            pos_gt = np.array(result_markers[ii][marker])
            M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix(pos_gt).T))
            M_model = human_data.oMf[human_model.getFrameId(marker)]
            pos_model = np.array(M_model.translation).flatten()

            M_model_frame[f"{marker}_x"] = M_model.translation[0]
            M_model_frame[f"{marker}_y"] = M_model.translation[1]
            M_model_frame[f"{marker}_z"] = M_model.translation[2]

            place(viz, 'world/' + marker, M)
            place(viz, 'world/' + marker + "_m", M_model)

            sq_error = np.sum((pos_gt - pos_model) ** 2)
            rmse_per_marker.setdefault(marker, []).append(sq_error)

        M_model_list.append(M_model_frame)
        viz.display(q)
        ik_class._q0 = q
        q_list.append(q)

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
    

    if len(joint_angles_names) != len(q_list[0]):
        raise ValueError("Mismatch between joint names and q size")

    pd.DataFrame(q_list, columns=joint_angles_names).to_csv(
        os.path.join(rt_cosmik_path, f"output/{no_trial}/mouv/{task}/q_mocap.csv"), index=False)

    print("\nPer-marker RMSE (in meters):")
    rmse_global, nb_mks = 0, 0
    for marker, sq_errors in rmse_per_marker.items():
        nb_mks += 1
        rmse = np.sqrt(np.mean(sq_errors))
        print(f"{marker}: {rmse:.4f} m")
        rmse_global += rmse

    print(f"Global RMSE: {rmse_global / nb_mks:.4f} m")


if __name__ == "__main__":
    task_list = ["bolting","bolting_sat","crouch","crouch_object","hitting","hitting_sat","jump","lifting","lifting_fast","lower","overhead"
             "robot_sanding","robot_welding",
             "sanding","sanding_sat","sit_to_stand","squat","static","upper","walk","walk_front","welding","welding_sat"]

    for task in task_list:
        run_ik(task)
