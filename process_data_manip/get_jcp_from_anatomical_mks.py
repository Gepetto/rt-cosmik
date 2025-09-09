import numpy as np
from  src.rtcosmik.utils.read_write_utils  import read_mks_data,udp_csv_to_dataframe
import pandas as pd
from src.rtcosmik.utils.linear_algebra_utils import transform_to_local_frame,transform_to_global_frame
from src.rtcosmik.human_model.model_utils import get_pelvis_pose
import os
subjects = [
    "4279"
]
tasks = ["robot_welding"]
gender = 'male'

mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

jcp_names = [
        "LShoulder", "RShoulder", "Neck",  "RElbow", "LElbow", 
        "RWrist", "LWrist", "RHip", "LHip", "midHip",

        "RKnee", "LKnee", "RAnkle", "LAnkle","RHeel", "LHeel",
         "RBigToe", "LBigToe", "RSmallToe", "LSmallToe"
    ]

def midpoint(p1, p2):
    return 0.5 * (np.array(p1) + np.array(p2))

def compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS, knee_study, ankle_study, side="right"):
    """
    Compute hip joint center using Leardini et al. (1999) method.
    
    """
    ASIS_mid = midpoint(R_ASIS, L_ASIS)
    PSIS_mid = midpoint(R_PSIS, L_PSIS)

    # Distance between ASIS and PSIS centers
    pelvis_depth_vec = ASIS_mid - PSIS_mid
    pelvis_depth = np.linalg.norm(pelvis_depth_vec)

    # Distance between ASIS markers (pelvis width)
    pelvis_width = np.linalg.norm(R_ASIS - L_ASIS)

    ankle_knee_length = np.linalg.norm(ankle_study - knee_study)
    knee_ASIS_length = np.linalg.norm(knee_study - (R_ASIS if side == "right" else L_ASIS))
    vertical_adjust = ankle_knee_length + knee_ASIS_length

    hip_y = ASIS_mid[1] - 0.096 * vertical_adjust
    # Compute hip center
    hip_x = ASIS_mid[0] - 0.31 * pelvis_depth
    if side == "right":        
        hip_z = ASIS_mid[2] + 0.38 * pelvis_width
    elif side == "left":
        hip_z = ASIS_mid[2] - 0.38 * pelvis_width
    else:
        raise ValueError("Side must be 'right' or 'left'")

    return np.array([hip_x, hip_y, hip_z])

def compute_uptrunk(C7, CLAV):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 8 * np.pi / 180
    return np.array([
        C7[0] + np.cos(angle_rad) * 0.55 * norm,
        C7[1] + np.sin(angle_rad) * 0.55 * norm,
        C7[2]
    ])

def compute_shoulder(SHO, C7, CLAV, side='right'):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 11 * np.pi / 180
    sign = -1 if side == 'right' else -1  # both use minus sign in paper

    return np.array([
        SHO[0] + np.cos(angle_rad) * 0.43 * norm,
        SHO[1] + sign * np.sin(angle_rad) * 0.43 * norm,
        SHO[2]
    ])


def compute_joint_centers_from_mks(markers):
    jcp = {}

    pelvis_pose = get_pelvis_pose(markers, gender=gender)
    pelvis_position = pelvis_pose[:3, 3].reshape(3, 1)
    pelvis_rotation = pelvis_pose[:3, :3]

    #  Transform all markers into pelvis (local) frame 
    markers_local = {}
    for name, coords in markers.items():
        coords = coords.reshape(3, 1)
        markers[name] = transform_to_local_frame(coords, pelvis_position, pelvis_rotation)

     #  Shoulders and Neck 
    try:
        jcp['RShoulder'] = compute_shoulder(markers['r_shoulder_study'], markers['C7_study'], markers['SJN'], 'right')
        jcp['LShoulder'] = compute_shoulder(markers['L_shoulder_study'], markers['C7_study'], markers['SJN'], 'left')
        jcp['Neck'] = compute_uptrunk(markers['C7_study'], markers['SJN'] )
    except KeyError:
        pass

        #  Elbows 
    try:
        jcp['RElbow'] = midpoint(markers['r_melbow_study'], markers['r_lelbow_study'])
        jcp['LElbow'] = midpoint(markers['L_melbow_study'], markers['L_lelbow_study'])
    except KeyError:
        pass

    #  Wrists 
    try:
        jcp['RWrist'] = midpoint(markers['r_mwrist_study'], markers['r_lwrist_study'])
        jcp['LWrist'] = midpoint(markers['L_mwrist_study'], markers['L_lwrist_study'])
    except KeyError:
        pass

    #  Pelvis and Hips 
    try:
        R_ASIS = markers['r.ASIS_study']
        L_ASIS = markers['L.ASIS_study']
        R_PSIS = markers['r.PSIS_study']
        L_PSIS = markers['L.PSIS_study']
    
        jcp['RHip'] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS, markers['r_knee_study'], markers['r_ankle_study'], side="right")
        jcp['LHip'] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS, markers['L_knee_study'], markers['L_ankle_study'], side="left")

        jcp['midHip']= midpoint(jcp['RHip'],jcp['LHip'])
    except KeyError:
        pass


    #  Knees 
    try:
        jcp['RKnee'] = midpoint(markers['r_mknee_study'], markers['r_knee_study'])
        jcp['LKnee'] = midpoint(markers['L_mknee_study'], markers['L_knee_study'])
    except KeyError:
        pass

    #  Ankles 
    try:
        jcp['RAnkle'] = midpoint(markers['r_mankle_study'], markers['r_ankle_study'])
        jcp['LAnkle'] = midpoint(markers['L_mankle_study'], markers['L_ankle_study'])
    except KeyError:
        pass
        
        #  Feet / Toes 
    try:
        jcp['RHeel'] = markers['r_calc_study']
        jcp['LHeel'] = markers['L_calc_study']
    except KeyError:
        pass

    try:
        jcp['RBigToe'] = markers['r_toe_study']
        jcp['LBigToe'] = markers['L_toe_study']
    except KeyError:
        pass

    try:
        jcp['RSmallToe'] = markers['r_5meta_study']
        jcp['LSmallToe'] = markers['L_5meta_study']
    except KeyError:
        pass

    jcp_global = {}
    for name, coords in jcp.items():
        coords = np.asarray(coords)
        if coords.shape == (3,):
            coords = coords.reshape(3,1)
        elif coords.shape == (3,3):
            print(f"⚠️ Skipping '{name}' – got unexpected shape {coords.shape} (likely a matrix)")
            continue
        global_coords = transform_to_global_frame(coords, pelvis_position, pelvis_rotation)
        jcp_global[name] = global_coords.flatten()

    return jcp_global
for no_trial in subjects:
    for task in tasks:
        base_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}"
        path_to_csv =f"{base_path}/mocap/{task}/mocap_downsampled_to_40hz.csv"
         # Skip if file doesn't exist
        if not os.path.exists(path_to_csv):
            print(f"Skipping missing task: {no_trial} / {task}")
            continue
        
        df = pd.read_csv(path_to_csv)
        # df = udp_csv_to_dataframe(path_to_csv, mks_names)
        df.columns = [col.replace(f"{no_trial}:", "") for col in df.columns]
        frames = df["Frame"] if "Frame" in df.columns else range(len(df))
        mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))

        mks_dict, start_sample_dict = read_mks_data(df, start_sample=0)

        jcp_per_frame = []
        for frame_id in range(len(mks_dict)):
            markers_frame = mks_dict[frame_id]
            jcp = compute_joint_centers_from_mks(markers_frame)
            jcp_per_frame.append(jcp)

        jcp_rows = []
        for jcp in jcp_per_frame:
            flat_jcp = {}
            for name, coords in jcp.items():
                flat_jcp[f"{name}_x"] = coords[0]
                flat_jcp[f"{name}_y"] = coords[1]
                flat_jcp[f"{name}_z"] = coords[2]
            jcp_rows.append(flat_jcp)

        jcp_df = pd.DataFrame(jcp_rows)
        path = f"{base_path}/mocap/{task}"
        os.makedirs(path, exist_ok=True)

        output_csv_path = f"{path}/joint_center_positions.csv"

        jcp_df.to_csv(output_csv_path, index=False)