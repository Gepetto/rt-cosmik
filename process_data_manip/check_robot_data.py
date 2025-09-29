import example_robot_data as robex 
import cv2
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
from pinocchio.visualize import GepettoVisualizer
import pinocchio as pin 
import numpy as np 
import sys
from src.rtcosmik.utils.read_write_utils import read_mks_data,udp_csv_to_dataframe,marker_data_to_dataframe,read_subject_info
import pandas as pd
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length
from src.rtcosmik.ik.ik import RT_IK
import yaml

import time 

def se3_to_yaml(se3: pin.SE3, path: str, key: str = "world_T_robot"):
    """
    Save a Pinocchio SE3 to YAML with multiple useful representations.
    """
    # Build 4x4 homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = se3.rotation
    T[:3, 3]  = se3.translation

    # Quaternion as (w, x, y, z) using Pinocchio helper
    xyzquat = pin.SE3ToXYZQUAT(se3)  # [x y z qw qx qy qz]
    qw, qx, qy, qz = map(float, xyzquat[3:7])

    data = {
        key: {
            "translation": se3.translation.tolist(),            # [x, y, z]
            "rotation_matrix": se3.rotation.tolist(),           # 3x3
            "quaternion": {"w": qw, "x": qx, "y": qy, "z": qz}, # unit quaternion
            "matrix_4x4": T.tolist(),                           # 4x4
            "convention": {
                "quaternion": "w,x,y,z",
                "matrix_order": "row-major",
                "meaning": f"{key} maps points from 'robot' to 'world' (X_world = T * X_robot)"
            }
        }
    }

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

subject_ids = {
    # "Nicolas": 2307,
    # "Mohamed": 1602,
    # "Clement": 1118,
    # "Mathis": 3361,
    # "Claire_": 4827,
    # "Anais": 4687,
    # "Emmanuelle": 4801,
    # "Maxime_": 1847,
    # "Alessandro": 4279,
    # "Marie_M": 2112,
    # "Anastasia": 4216,
    # "Flavie": 1012,
    "Zoe": 4162,
#     "Kahina": 4665,
#     "Herbert": 1508,
#     "Guilhem": 4509,
#     "Bilal": 4612,
#     "Batiste": 2198
}

SUBJECTS = subject_ids.keys()


for subject in SUBJECTS:
    print(f"Processing subject {subject}...")
    info_path = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/info.txt"
    subject_height,subject_mass, gender = read_subject_info(info_path)

    rt_cosmik_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/robot_sanding/mks_data_gapfilled.csv"):
        path_to_csv = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/robot_sanding/mks_data_gapfilled.csv"
    else:
        pass
        
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

    # Load UDP CSV (wide format)
    df_wide = udp_csv_to_dataframe(path_to_csv, mks_names)
    # df_wide = pd.read_csv(path_to_csv)
    # df_wide.columns = [col.replace(f"{no_trial}:", "") for col in df_wide.columns]
    # frames = df_wide["Frame"] if "Frame" in df_wide.columns else range(len(df_wide))
    # mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df_wide.columns if "_x" in col))
    result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=0, converter = 1.0)

    # Load URDF
    human = Robot('/home/msabbah/pinocchio-3x/src/rt-cosmik/urdf/human.urdf', rt_cosmik_path, isFext=True)
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    human_model = scale_human_model(human_model, start_sample_dict, with_hand=True, gender=gender, subject_height=subject_height)
    human_model = mks_registration(human_model, start_sample_dict, with_hand=True)
    human_data = pin.Data(human_model)

    #########################################################LOCK JOINTS 
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
    #######################################################################################

    # Load human kinematics
    if os.path.exists(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/robot_sanding/q_mocap.csv"):
        q_data = pd.read_csv(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/robot_sanding/q_mocap.csv").to_numpy()
    else: 
        q_data = pd.read_csv(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/robot_sanding/q_mocap_downsampled.csv").to_numpy()
    # q_data = q_data[159:, :]

    viz = GepettoVisualizer(human_model, human_visual_model,human_visual_model)
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

    # for ii in range(q_data.shape[0]):
    #     q = pin.neutral(human_model)
    #     q[:] = q_data[ii, :]
    #     viz.display(q)
    #     input()

    q = pin.neutral(human_model)
    q[:] = q_data[0, :]
    viz.display(q)

    # Load panda model
    robot = robex.load("panda")

    # Create a list of joints to lock
    jointsToLock = ['panda_finger_joint1', 'panda_finger_joint2']

    # Get the ID of all existing joints
    jointsToLockIDs = []
    for jn in jointsToLock:
        if robot.model.existJointName(jn):
            jointsToLockIDs.append(robot.model.getJointId(jn))
        else:
            print('Warning: joint ' + str(jn) + ' does not belong to the model!')

    initialJointConfig = pin.neutral(robot.model)

    geom_models = [robot.visual_model, robot.collision_model]
    model_reduced, geometric_models_reduced = pin.buildReducedModel(robot.model,list_of_geom_models=geom_models,list_of_joints_to_lock=jointsToLockIDs,reference_configuration=initialJointConfig)

    # geometric_models_reduced is a list, ordered as the passed variable "geom_models" so:
    visual_model_reduced, collision_model_reduced = geometric_models_reduced[0], geometric_models_reduced[1]

    # Get markers of the base 
    markers_base = pd.read_csv(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/robot_in_world/{subject_ids[subject]}/robot_position_trajectories.csv")

    # Get position of the base 
    # marker_robot2 = np.array([770.454224,2.117334,950.284363])*1e-3 # haut droite
    # marker_robot1 = np.array([983.845886,-4.049469,955.955017])*1e-3 # haut gauche
    # marker_robot3 = np.array([925.026306,156.854385,955.778748])*1e-3 # bas gauche
    # marker_robot4 = np.array([830.677856,156.463226,955.619507])*1e-3 # bas droit
    
    # if subject =='Zoe':
    #     marker_robot2 = np.array([-markers_base['robot_base_segment1_y'][0], markers_base['robot_base_segment1_z'][0], -markers_base['robot_base_segment1_x'][0]])*1e-3 # haut droite
    #     marker_robot1 = np.array([-markers_base['robot_base_segment2_y'][0], markers_base['robot_base_segment2_z'][0], -markers_base['robot_base_segment2_x'][0]])*1e-3 # haut gauche
    #     marker_robot3 = np.array([-markers_base['robot_base_segment3_y'][0], markers_base['robot_base_segment3_z'][0], -markers_base['robot_base_segment3_x'][0]])*1e-3 # bas gauche
    #     marker_robot4 = np.array([-markers_base['robot_base_segment4_y'][0], markers_base['robot_base_segment4_z'][0], -markers_base['robot_base_segment4_x'][0]])*1e-3 # bas droit
    # else: 
    
    marker_robot2 = np.array([markers_base['robot_base_segment1_x'][0], markers_base['robot_base_segment1_y'][0], markers_base['robot_base_segment1_z'][0]])*1e-3 # haut droite
    marker_robot1 = np.array([markers_base['robot_base_segment2_x'][0], markers_base['robot_base_segment2_y'][0], markers_base['robot_base_segment2_z'][0]])*1e-3 # haut gauche
    marker_robot3 = np.array([markers_base['robot_base_segment3_x'][0], markers_base['robot_base_segment3_y'][0], markers_base['robot_base_segment3_z'][0]])*1e-3 # bas gauche
    marker_robot4 = np.array([markers_base['robot_base_segment4_x'][0], markers_base['robot_base_segment4_y'][0], markers_base['robot_base_segment4_z'][0]])*1e-3 # bas droit

    markers_robot = [marker_robot1, marker_robot2, marker_robot3, marker_robot4]

    for ii in range(4):
        viz.viewer.gui.addSphere(f"world/robot_marker_{ii+1}",0.01,[1,0,0,1])
        place(viz, f"world/robot_marker_{ii+1}", pin.SE3(np.eye(3), np.matrix([markers_robot[ii][0], markers_robot[ii][1], markers_robot[ii][2]]).T))

    P1 = robot_center = (marker_robot1+marker_robot2)/2
    P2 = (marker_robot3+marker_robot4)/2
    P3 = marker_robot1

    P2P1 = P1-P2
    P1P3 = P3-P1

    Vz = np.cross(P2P1, P1P3)
    Vy = np.cross(Vz,P2P1)
    Vx = P2P1

    x_axis = Vx/np.linalg.norm(Vx)
    y_axis = Vy/np.linalg.norm(Vy)
    z_axis = Vz/np.linalg.norm(Vz)

    # 1. Construct the rotation matrix
    robot_orientation = np.column_stack((x_axis, y_axis, z_axis))

    SE3_base_robot = pin.SE3(robot_orientation, robot_center)

    se3_to_yaml(SE3_base_robot, f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/robot_in_world/{subject_ids[subject]}/robot_base_pose.yaml", key="world_T_robot")

    viz.viewer.gui.addXYZaxis('world/robot_base', [0, 255., 0, 1.], 0.04, 0.2)
    place(viz, 'world/robot_base', SE3_base_robot)

    # Find the root joint (parent == 0). For you this is 1.
    root_id = next(j for j in range(1, model_reduced.njoints) if model_reduced.parents[j] == 0)

    # Pre-multiply to keep any URDF offset
    model_reduced.jointPlacements[root_id] = SE3_base_robot * model_reduced.jointPlacements[root_id]

    visual_model_reduced.geometryObjects.tolist()[0].placement = SE3_base_robot

    viz_robot = GepettoVisualizer(model_reduced, collision_model_reduced, visual_model_reduced)
    viz_robot.initViewer()
    viz_robot.loadViewerModel("panda")

    q_panda = pin.neutral(model_reduced)
    viz_robot.display(q_panda)
    input("Press Enter to go next...")

# CSV = "/home/msabbah/pinocchio-3x/src/rt-cosmik/output/robot/cam0_robot_interpolated.csv"

# # --- Load & compute deltas (keep tz-aware UTC) ---
# df = pd.read_csv(CSV)
# # Prefer _cam_time (what we wrote when interpolating)
# time_col = "_cam_time" if "_cam_time" in df.columns else next(
#     c for c in df.columns if c.lower() in {"timestamp","time","datetime","stamp"}
# )

# df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
# df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

# # Δt in seconds between consecutive frames
# timestamp_deltas = df[time_col].diff().dt.total_seconds().to_numpy()
# timestamp_deltas[0] = 0.0
# # Be safe: no negatives / NaNs
# timestamp_deltas = np.clip(np.nan_to_num(timestamp_deltas, nan=0.0), 0.0, None)

# # --- Grab robot joint data (adapt slice to your CSV layout) ---
# # Your example uses columns 2..8 for robot q:
# robot_q_data = df.iloc[:, 2:9].to_numpy(dtype=float)

# # Optional: speed factor (e.g., 0.5 = half speed, 2.0 = double speed)
# speed = 1.0
# timestamp_deltas = timestamp_deltas / max(speed, 1e-9)

# input("Ready to start the recording ?")

# # --- Playback without drift ---
# next_t = time.perf_counter()  # monotonic start

# for ii in range(robot_q_data.shape[0]):
#     # update your two models
#     q[:] = q_data[ii, :]         # your human/other model (as in your snippet)
#     viz.display(q)

#     q_panda[:] = robot_q_data[ii, :]
#     viz_robot.display(q_panda)

#     time.sleep(1/40)
#     # # schedule next frame based on absolute time, not cumulative sleeps
#     # next_t += timestamp_deltas[ii]
#     # while True:
#     #     remaining = next_t - time.perf_counter()
#     #     if remaining <= 0:
#     #         break
#     #     time.sleep(min(remaining, 0.005))  # small sleeps to stay responsive

