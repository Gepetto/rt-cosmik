#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from typing import List, Tuple, Optional

# --- Make top-level 'src' importable for both main and workers ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../pinocchio-3x
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------- Your existing imports & code ----------------
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

from src.rtcosmik.human_model.urdf_model import *
from src.rtcosmik.utils.read_write_utils import read_mks_data, udp_csv_to_dataframe, read_subject_info
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.human_model.model_utils import get_segment_length
from src.rtcosmik.ik.ik import RT_IK

SUBJECTS = [
    "Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie","Guilhem",
    "Kahina","Marie_M","Mathis","Maxime_","Mohamed","Nicolas","Zoe","Herbert","Emmanuelle"
]

TASKS = [
    "bolting","bolting_sat","crouch","crouch_object","hitting","hitting_sat","jump","lifting",
    "lifting_fast","lower","overhead","overhead_front","robot_sanding","robot_welding",
    "sanding","sanding_sat","sit_to_stand","squat","static","upper","walk","walk_front",
    "welding","welding_sat"
]


def run_ik(task, no_trial, start_sample=0, visualize=False):
    info_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/info.txt"
    subject_height,subject_mass, gender = read_subject_info(info_path)

    rt_cosmik_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/mocap_downsampled_to_40hz.csv"
    df_wide = pd.read_csv(path_to_csv)

    mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis',
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

    # Load UDP CSV (wide format)
    if not os.path.exists(path_to_csv):

        raise FileNotFoundError(f"Missing CSV: {path_to_csv}")
    # df_wide = udp_csv_to_dataframe(path_to_csv, mks_names)
    result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=start_sample, converter = 1000.0)

    # Load URDF
    human = Robot('/root/workspace/ros_ws/src/rt-cosmik/urdf/human.urdf', rt_cosmik_path, isFext=True)
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    human_model = scale_human_model(human_model, start_sample_dict, with_hand=True, gender=gender, subject_height=subject_height)
    human_model = mks_registration(human_model, start_sample_dict, with_hand=True)
    human_data = pin.Data(human_model)

    # --- LOCK JOINTS ---
    # joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
    # joint_ids_to_lock = []
    # for jn in joints_to_lock:
    #     if human_model.existJointName(jn):
    #         joint_ids_to_lock.append(human_model.getJointId(jn))
    #     else:
    #         print('Warning: joint ' + str(jn) + ' does not belong to the model!')

    # q0 = pin.neutral(human_model)
    # human_model, human_visual_model = pin.buildReducedModel(human_model, human_visual_model, joint_ids_to_lock, q0)
    # human_data = pin.Data(human_model)

    # --- Visualization (optional) ---
    if visualize:
        viz = gv_init(human_model, human_collision_model, human_visual_model, start_sample_dict)
        pin.forwardKinematics(human_model, human_data, pin.neutral(human_model))
        pin.updateFramePlacements(human_model, human_data)

        for frame in human_model.frames.tolist():
            viz.viewer.gui.addXYZaxis('world/' + frame.name, [1, 0, 0, 1], 0.01, 0.1)
            place(viz, 'world/' + frame.name, human_data.oMf[human_model.getFrameId(frame.name)])

    q = pin.neutral(human_model)
    human_data = pin.Data(human_model)
    if visualize:
        viz.display(q)

    if visualize:
        seg_frames = construct_segments_frames(result_markers[start_sample])
        add_frames(viz, seg_frames, "meas", 0.008, 0.08)
        add_marker(viz, result_markers[1].keys(), '_m', 0, 0, 1)
        for joint_id in range(1, human_model.njoints):
            frame_name = f'world/{human_model.names[joint_id] + "_model"}'
            viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], 0.012, 0.05)

    keys_to_track_list = [
        'BHD','RHD','LHD','FHD',
        'C7_study','TV8','TV12','SJN','STRN',
        'r.ASIS_study', 'L.ASIS_study',
        'r.PSIS_study', 'L.PSIS_study',
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study', 'RHL2','RHM5',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study',
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study','LHL2','LHM5',
        'L_ankle_study', 'L_mankle_study',
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study'
    ]

    ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt=1/100)
    q = ik_class.solve_ik_sample_casadi()
    if visualize:
        viz.display(q)
    ik_class._q0 = q

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

            M_model_frame[f"{marker}_x"] = float(M_model.translation[0])
            M_model_frame[f"{marker}_y"] = float(M_model.translation[1])
            M_model_frame[f"{marker}_z"] = float(M_model.translation[2])

            if visualize:
                place(viz, 'world/' + marker, M)
                place(viz, 'world/' + marker + "_m", M_model)

            sq_error = float(np.sum((pos_gt - pos_model) ** 2))
            rmse_per_marker.setdefault(marker, []).append(sq_error)

        M_model_list.append(M_model_frame)
        if visualize:
            viz.display(q)
        ik_class._q0 = q
        q_list.append(q)

    # save mks est (model markers)
    df = pd.DataFrame(M_model_list)
    # FIX: os.path.join with an absolute path argument discards the prefix; use the absolute path directly.
    csv_file = f"/root/workspace/ros_ws/src/rt-cosmik/output/mocap/{no_trial}/{task}/joint_angles.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    df.to_csv(csv_file, index=False)

    joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'Thoracic_flex_ext','Thoracic_lateral_flex','Thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi','Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']

    if len(joint_angles_names) != len(q_list[0]):
        raise ValueError("Mismatch between joint names and q size")

    out_q = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/q_mocap_downsampled.csv"
    os.makedirs(os.path.dirname(out_q), exist_ok=True)
    pd.DataFrame(q_list, columns=joint_angles_names).to_csv(out_q, index=False)

    # Optional: return RMSE summary for logging
    rmse_global, nb_mks = 0.0, 0
    for marker, sq_errors in rmse_per_marker.items():
        nb_mks += 1
        rmse = float(np.sqrt(np.mean(sq_errors)))
        rmse_global += rmse
    global_rmse = float(rmse_global / max(nb_mks, 1))
    return global_rmse


def process_one_subject(no_trial: str, tasks: List[str], start_sample: int, visualize: bool
                        ) -> Tuple[str, int, int]:
    """
    Runs all tasks for a single subject, sequentially.
    Returns: (subject, n_success, n_fail)
    """
    ok = ko = 0
    for task in tasks:
        try:
            print(f"\n=== Subject: {no_trial} | Task: {task} ===")
            rmse = run_ik(task, no_trial=no_trial, start_sample=start_sample, visualize=visualize)
            print(f"[OK]   {no_trial} / {task} | Global RMSE: {rmse:.4f} m")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {no_trial} / {task} | {type(e).__name__}: {e}")
            ko += 1
    return no_trial, ok, ko


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel IK by subject (tasks run serially per subject)")
    parser.add_argument("--subjects", nargs="*", default=SUBJECTS, help="Subset of subjects to run")
    parser.add_argument("--tasks",    nargs="*", default=TASKS,    help="Subset of tasks to run for each subject")
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", help="Enable viewer (use --workers 1 if you turn this on)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers = subjects in flight (default: half CPU)")
    args = parser.parse_args()

    # Use half your cores by default, but not more than the number of subjects you asked for
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    max_workers = args.workers if args.workers > 0 else min(default_workers, len(args.subjects))
    if args.visualize and max_workers > 1:
        print("[WARN] Visualization with >1 worker can be unstable. Consider --workers 1.")

    print(f"Launching {len(args.subjects)} subject(s) with {max_workers} worker(s). "
          f"Each subject will run {len(args.tasks)} task(s) serially. "
          f"(visualize {'ON' if args.visualize else 'OFF'})")

    total_ok = total_ko = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_subject, subj, args.tasks, args.start_sample, args.visualize): subj
                   for subj in args.subjects}
        for fut in as_completed(futures):
            subj = futures[fut]
            try:
                _, ok, ko = fut.result()
                total_ok += ok
                total_ko += ko
                print(f"[SUMMARY] {subj}: {ok} OK / {ko} FAIL")
            except Exception as e:
                print(f"[WORKER FAIL] {subj}: {type(e).__name__}: {e}")

    print(f"\nDone. Overall: {total_ok} OK / {total_ko} FAIL")