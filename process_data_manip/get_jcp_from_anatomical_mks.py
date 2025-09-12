import numpy as np
from  src.rtcosmik.utils.read_write_utils  import read_mks_data,udp_csv_to_dataframe
import pandas as pd
from src.rtcosmik.utils.linear_algebra_utils import transform_to_local_frame,transform_to_global_frame
from src.rtcosmik.human_model.model_utils import get_pelvis_pose, get_torso_pose,get_virtual_pelvis_pose
import os
import matplotlib.pyplot as plt


subjects = [
    "2307","1602","1118","3361","4827","4687","4801","1847","4279","2112","4216","1012","4162","4665","1508","4509","4612","2198"
]
tasks = ["robot_welding"]
gender = 'male'

# mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
#              'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
#              'BHD','RHD','LHD','FHD',
#              'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
#              'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
#              'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
#              'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
#              'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
#              'r_pelvis', 'l_pelvis']

jcp_names = [
        "RShoulder", "LShoulder", "Neck",  "RElbow", "LElbow", 
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

def compute_shoulder(SHO, C7, CLAV):
    return np.array([
        SHO[0] ,#+ np.cos(11 * np.pi / 180) * 0.43  * np.linalg.norm(CLAV - C7),
        SHO[1] - np.sin(11 * np.pi / 180) * 0.43 * np.linalg.norm(CLAV - C7),
        SHO[2]
    ])

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)

# def compute_shoulder_(RSHO,LSHO, torso_pose):
#     bi_acromial_dist = np.linalg.norm(LSHO - RSHO)
    
#     rshoulder_center = RSHO.reshape(3,1) + 
#     lshoulder_center = LSHO.reshape(3,1) + 
#     return rshoulder_center, lshoulder_center

def compute_joint_centers_from_mks(markers, *, units="mm"):
    """
    Compute joint center positions and segment lengths from marker positions.

    Parameters
    ----------
    markers : dict[str, np.ndarray]
        Dict of global marker positions. Each value should be shape (3,) or (3,1),
        in either millimeters ("mm") or meters ("m") depending on `units`.
    units : {"mm", "m"}, optional
        Input units for `markers`. Used only for reporting lengths (meters).

    Returns
    -------
    jcp_global : dict[str, np.ndarray]
        Joint centers in GLOBAL frame, each as 1D array shape (3,) in input units.
    segment_lengths : dict[str, float]
        Upper/lower arm segment lengths in meters.
    norms : dict[str, list[float]]
        Elbow inter-epicondyle distances in meters. (Lists so you can append per-frame upstream.)
    """
    # --- helpers ---
    def as_col(x):
        x = np.asarray(x)
        return x.reshape(3, 1) if x.shape != (3, 1) else x

    mm_to_m = 0.001 if units == "mm" else 1.0

    jcp = {}
    norms = {"RElbow": [], "LElbow": []}

    # Pelvis pose (global)
    pelvis_pose = get_virtual_pelvis_pose(markers)
    pelvis_position = as_col(pelvis_pose[:3, 3])
    pelvis_rotation = pelvis_pose[:3, :3]

    bi_acromial_dist = bi_acromial_dist = np.linalg.norm(markers['L_shoulder_study'] - markers['r_shoulder_study'])
    torso_pose = get_torso_pose(markers)
    trans_global = (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)

    trans_local = transform_to_local_frame(trans_global,pelvis_position,pelvis_rotation)

    # ---- Transform all markers into pelvis (local) frame (do NOT mutate input) ----
    markers_local = {}
    for name, coords in markers.items():
        coords_col = as_col(coords)
        markers_local[name] = transform_to_local_frame(coords_col, pelvis_position, pelvis_rotation)

    # ---- Shoulders & Neck ----
    try:
        # jcp["RShoulder"] = compute_shoulder(markers_local["r_shoulder_study"],
        #                                     markers_local["C7_study"],
        #                                     markers_local["SJN"])
        # jcp["LShoulder"] = compute_shoulder(markers_local["L_shoulder_study"],
        #                                     markers_local["C7_study"],
        #                                     markers_local["SJN"])

        jcp["RShoulder"]= markers_local["r_shoulder_study"] 
        jcp["LShoulder"] = markers_local["L_shoulder_study"]
        jcp["Neck"] = compute_uptrunk(markers_local["C7_study"], markers_local["SJN"])
    except KeyError as e:
        # Missing any of these markers → skip shoulders/neck
        pass

    # ---- Elbows ----
    try:
        jcp["RElbow"] = midpoint(markers_local["r_melbow_study"], markers_local["r_lelbow_study"])
        jcp["LElbow"] = midpoint(markers_local["L_melbow_study"], markers_local["L_lelbow_study"])

        vec_r = markers_local["r_lelbow_study"] - markers_local["r_melbow_study"]
        vec_l = markers_local["L_lelbow_study"] - markers_local["L_melbow_study"]
        norms["RElbow"].append(np.linalg.norm(vec_r) * mm_to_m)
        norms["LElbow"].append(np.linalg.norm(vec_l) * mm_to_m)
    except KeyError:
        pass

    # ---- Wrists ----
    try:
        jcp["RWrist"] = midpoint(markers_local["r_mwrist_study"], markers_local["r_lwrist_study"])
        jcp["LWrist"] = midpoint(markers_local["L_mwrist_study"], markers_local["L_lwrist_study"])
    except KeyError:
        pass

    # ---- Pelvis & Hips ----
    try:
        R_ASIS = markers_local["r.ASIS_study"]
        L_ASIS = markers_local["L.ASIS_study"]
        R_PSIS = markers_local["r.PSIS_study"]
        L_PSIS = markers_local["L.PSIS_study"]

        jcp["RHip"] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS,
                                               markers_local["r_knee_study"],
                                               markers_local["r_ankle_study"],
                                               side="right")
        jcp["LHip"] = compute_hip_joint_center(L_ASIS, R_ASIS, L_PSIS, R_PSIS,
                                               markers_local["L_knee_study"],
                                               markers_local["L_ankle_study"],
                                               side="left")
        jcp["midHip"] = midpoint(jcp["RHip"], jcp["LHip"])
    except KeyError:
        pass

    # ---- Knees ----
    try:
        jcp["RKnee"] = midpoint(markers_local["r_mknee_study"], markers_local["r_knee_study"])
        jcp["LKnee"] = midpoint(markers_local["L_mknee_study"], markers_local["L_knee_study"])
    except KeyError:
        pass

    # ---- Ankles ----
    try:
        jcp["RAnkle"] = midpoint(markers_local["r_mankle_study"], markers_local["r_ankle_study"])
        jcp["LAnkle"] = midpoint(markers_local["L_mankle_study"], markers_local["L_ankle_study"])
    except KeyError:
        pass

    # ---- Feet / Toes ----
    try:
        jcp["RHeel"] = markers_local["r_calc_study"]
        jcp["LHeel"] = markers_local["L_calc_study"]
    except KeyError:
        pass

    try:
        jcp["RBigToe"] = markers_local["r_toe_study"]
        jcp["LBigToe"] = markers_local["L_toe_study"]
    except KeyError:
        pass

    try:
        jcp["RSmallToe"] = markers_local["r_5meta_study"]
        jcp["LSmallToe"] = markers_local["L_5meta_study"]
    except KeyError:
        pass

    # ---- Back to GLOBAL frame ----
    jcp_global = {}
    for name, coords in jcp.items():
        coords_col = as_col(coords)
        # Guard against accidental matrices (e.g., someone returns a 3x3)
        if coords_col.shape != (3,1):
            # try to coerce; if it fails, skip
            try:
                coords_col = np.asarray(coords).reshape(3,1)
            except Exception:
                print(f"⚠️ Skipping '{name}' – unexpected shape {np.asarray(coords).shape}")
                continue
        global_coords = transform_to_global_frame(coords_col, pelvis_position, pelvis_rotation)
        jcp_global[name] = global_coords.flatten()

    # ---- Segment lengths (in meters) ----
    segment_lengths = {}
    try:
        segment_lengths["RUpperArm"] = np.linalg.norm(jcp_global["RElbow"] - jcp_global["RShoulder"]) * mm_to_m
        segment_lengths["RLowerArm"] = np.linalg.norm(jcp_global["RWrist"] - jcp_global["RElbow"]) * mm_to_m
    except KeyError:
        pass

    try:
        segment_lengths["LUpperArm"] = np.linalg.norm(jcp_global["LElbow"] - jcp_global["LShoulder"]) * mm_to_m
        segment_lengths["LLowerArm"] = np.linalg.norm(jcp_global["LWrist"] - jcp_global["LElbow"]) * mm_to_m
    except KeyError:
        pass

    return jcp_global, segment_lengths, norms


all_segment_lengths = []
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

        all_norms_R = []
        all_norms_L = []

        jcp_per_frame = []
        seg_lengths_rows = []
        for frame_id in range(len(mks_dict)):
            markers_frame = mks_dict[frame_id]
            jcp, seg_lengths,norms = compute_joint_centers_from_mks(markers_frame)
            jcp_per_frame.append(jcp)
            if norms["RElbow"]:
                all_norms_R.extend(norms["RElbow"])
            if norms["LElbow"]:
                all_norms_L.extend(norms["LElbow"])

            row = {"Frame": frame_id}
            for k, v in seg_lengths.items():
                row[k] = v
            seg_lengths_rows.append(row)
        
        seg_df = pd.DataFrame(seg_lengths_rows)
        seg_df["Subject"] = no_trial
        seg_df["Task"] = task

        all_segment_lengths.append(seg_df)

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

        output_csv_path = f"{path}/joint_center_positions_test.csv"

        jcp_df.to_csv(output_csv_path, index=False)

        fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)

        axes[0].plot(all_norms_R, label="Right Elbow", color="blue")
        axes[0].set_ylabel("Norm (m)")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(all_norms_L, label="Left Elbow", color="red")
        axes[1].set_ylabel("Norm (m)")
        axes[1].set_xlabel("Frame")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        # fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # # Right upper arm
        # axes[0].plot(seg_df["Frame"], seg_df["RUpperArm"], color="blue")
        # axes[0].set_ylabel("R Upper Arm (m)")
        # axes[0].grid(True)
        # axes[0].set_title(f"{no_trial} - {task} : Segment Lengths")

        # # Right lower arm
        # axes[1].plot(seg_df["Frame"], seg_df["RLowerArm"], color="cyan")
        # axes[1].set_ylabel("R Lower Arm (m)")
        # axes[1].grid(True)

        # # Left upper arm
        # axes[2].plot(seg_df["Frame"], seg_df["LUpperArm"], color="red")
        # axes[2].set_ylabel("L Upper Arm (m)")
        # axes[2].grid(True)

        # # Left lower arm
        # axes[3].plot(seg_df["Frame"], seg_df["LLowerArm"], color="orange")
        # axes[3].set_ylabel("L Lower Arm (m)")
        # axes[3].set_xlabel("Frame")
        # axes[3].grid(True)

        # plt.tight_layout()
        # plt.show()