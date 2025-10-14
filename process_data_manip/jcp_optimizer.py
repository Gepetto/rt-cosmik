#!/usr/bin/env python3

# to launch:
# run process_data_manip/jcp_optimizer.py --subject-name Alessandro --subject-id 4279 --task static --mocap-csv
#  /home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/static/mocap_downsampled_to_40hz.csv --out
# /home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/static/optimized_augmented_markers.csv --sav
# e-offsets-json /home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/static/optimized_offset.json
# --mass 73 --height 1.87 --viz


import os, sys, json, time
from pathlib import Path
from collections import deque
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----------------- repo src path -----------------
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.joinpath("../../../src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----------------- project imports -----------------
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from src.rtcosmik.utils.read_write_utils import read_mks_data, save_to_csv
from src.rtcosmik.utils.linear_algebra_utils import transform_to_local_frame, transform_to_global_frame
from src.rtcosmik.human_model.model_utils import construct_segments_frames, get_torso_pose, get_virtual_pelvis_pose

# (Optional) minimal viz like your style; safe to keep
try:
    import pinocchio as pin
    from pinocchio.visualize import GepettoVisualizer
    HAVE_VIZ = True
except Exception:
    HAVE_VIZ = False

# --------------- names ----------------
JCP_NAMES = [
    "RShoulder","LShoulder","Neck",
    "RElbow","LElbow",
    "RWrist","LWrist",
    "RHip","LHip","midHip",
    "RKnee","LKnee",
    "RAnkle","LAnkle",
    "RHeel","LHeel",
    "RBigToe","LBigToe",
    "RSmallToe","LSmallToe",
]

AUGMENTED_MARKERS = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
    'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
    'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
    'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
    'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
    'L_lwrist_study','L_mwrist_study'
]

MOCAP_MARKERS = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
    'C7_study', 'r_lelbow_study',
    'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
    'L_lwrist_study','L_mwrist_study'
]

AUGMENTED_HEADER = [f"{m}_{axis}" for m in AUGMENTED_MARKERS for axis in ("x","y","z")]
MOCAP_HEADER = [f"{m}_{axis}" for m in MOCAP_MARKERS for axis in ("x","y","z")]


TO_DROP = {
    'r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study','L_thigh2_study','L_thigh3_study',
    'r_sh1_study','r_sh2_study','r_sh3_study','L_sh1_study','L_sh2_study','L_sh3_study', 'RHJC_study', 'LHJC_study'
}

DROP_IDX = [i for i, n in enumerate(AUGMENTED_MARKERS) if n in TO_DROP]  # row indices in (43,3)
keypoints_buffer = deque(maxlen=30)

# ---------- JCP utils ----------
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
        SHO[0] + np.cos(11 * np.pi / 180) * 0.43  * np.linalg.norm(CLAV - C7),
        SHO[1] - np.sin(11 * np.pi / 180) * 0.43 * np.linalg.norm(CLAV - C7),
        SHO[2]
    ])

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)

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
    jcp_g = {}
    norms = {"RElbow": [], "LElbow": []}

    # Pelvis pose (global)
    pelvis_pose = get_virtual_pelvis_pose(markers)
    pelvis_position = as_col(pelvis_pose[:3, 3])
    pelvis_rotation = pelvis_pose[:3, :3]

    bi_acromial_dist = np.linalg.norm(markers['L_shoulder_study'] - markers['r_shoulder_study'])
    torso_pose = get_torso_pose(markers)

    # ---- Transform all markers into pelvis (local) frame (do NOT mutate input) ----
    markers_local = {}
    for name, coords in markers.items():
        coords_col = as_col(coords)
        markers_local[name] = transform_to_local_frame(coords_col, pelvis_position, pelvis_rotation)

    # ---- Shoulders & Neck ----
    try:
        jcp_g["RShoulder"]= markers['r_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)
        jcp_g["LShoulder"] = markers['L_shoulder_study'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)

        jcp["RShoulder"] = transform_to_local_frame(jcp_g["RShoulder"], pelvis_position, pelvis_rotation)
        jcp["LShoulder"] = transform_to_local_frame(jcp_g["LShoulder"], pelvis_position, pelvis_rotation)

        ###pontonnier
        # jcp['RShoulder'] = compute_shoulder(markers_local['r_shoulder_study'], markers_local['C7_study'], markers_local['SJN'])
        # jcp['LShoulder'] = compute_shoulder(markers_local['L_shoulder_study'], markers_local['C7_study'], markers_local['SJN'])


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
                       # meters

def load_offsets_json(path, jcp_names=JCP_NAMES):
    """
    Returns:
      offs: dict[name] -> [x,y,z] (meters)
      off_mat: np.ndarray (20,3) in the order of jcp_names
      theta: np.ndarray (60,) flattened per-JCP offsets
    """
    with open(path, "r") as f:
        data = json.load(f)

    offs = data["offsets_per_jcp_m"]  # meters
    # build matrix in the exact order expected by your code
    off_mat = np.stack([np.asarray(offs.get(name, [0.0, 0.0, 0.0]), dtype=float)
                        for name in jcp_names], axis=0)  # (20,3)
    theta = off_mat.reshape(-1)  # (60,)
    return offs, off_mat, theta

# ----------------- visualization (optional; minimal) -----------------
def _place(viz, node, p3):
    se3 = pin.SE3(np.eye(3), np.asarray(p3, float).reshape(3,1))
    viz.viewer.gui.applyConfiguration(node, pin.SE3ToXYZQUAT(se3).tolist())

def visualize_simple(preds_full, meas_m_full, aug_markers_base, jcp_m_full, jcp_off_full, sleep_dt=0.02):
    if not HAVE_VIZ:
        print("[viz] Pinocchio/Gepetto not available.")
        return
    T = preds_full.shape[0]; J = len(JCP_NAMES); M = len(MOCAP_MARKERS)
    aug_vis  = preds_full.reshape(T,M,3)           # (T, M_have, 3)
    meas_vis = meas_m_full.reshape(T,M,3)                       # (T, M_have, 3)
    aug_markers_base_vis = aug_markers_base.reshape(T,M,3)       

    jcp0 = jcp_m_full.reshape(T, J, 3)
    jcp1 = jcp_off_full.reshape(T, J, 3)

    viz = GepettoVisualizer()
    viz.initViewer()
    viz.loadViewerModel("pinocchio")
    try:
        viz.viewer.gui.addXYZaxis('world/base', [255, 0, 0, 1.], 0.04, 0.2)
    except Exception:
        pass

    for name in MOCAP_MARKERS:
        viz.viewer.gui.addSphere(f'world/mocap_{name}', 0.015, [255, 0, 0, 1.])
        viz.viewer.gui.addSphere(f'world/aug_{name}',   0.015, [0, 0, 255, 1.])
        viz.viewer.gui.addSphere(f'world/aug_base_{name}',   0.015, [0, 0, 0, 1.])
    for jname in JCP_NAMES:
        viz.viewer.gui.addSphere(f'world/jcp0_{jname}', 0.012, [0, 255, 0, 1.])
        viz.viewer.gui.addSphere(f'world/jcp1_{jname}', 0.012, [255, 255, 0, 1.])

    for t in range(T):
        for m_i, name in enumerate(MOCAP_MARKERS):
            _place(viz, f'world/mocap_{name}', meas_vis[t, m_i])
            _place(viz, f'world/aug_{name}',   aug_vis[t,  m_i])
            _place(viz, f'world/aug_base_{name}',   aug_markers_base_vis[t,  m_i])
        for j_i, jname in enumerate(JCP_NAMES):
            _place(viz, f'world/jcp0_{jname}', jcp0[t, j_i])
            _place(viz, f'world/jcp1_{jname}', jcp1[t, j_i])
        viz.viewer.gui.refresh()
        time.sleep(sleep_dt)

# ----------------- main optimization -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Per-JCP offset optimization over the whole dataset (global RMSE).")
    # IO
    p.add_argument("--subject-name", required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--base-path", default="/home/msabbah/pinocchio-3x/src/rt-cosmik")
    p.add_argument("--mocap-csv", default=None)
    p.add_argument("--out", default=None, help="Output CSV (meters) for best augmented markers.")
    p.add_argument("--save-offsets-json", default=None, help="Where to save learned (20x3) offsets JSON.")
    # Augmenter
    p.add_argument("--mass", type=float, required=True)
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--augmenter-dir",
                   default="/home/msabbah/pinocchio-3x/src/rt-cosmik/src/rtcosmik/augmenter/augmentation_model")
    p.add_argument("--augmenter-model", default="v0.3")
    # Optimization (per-JCP only)
    p.add_argument("--lambda-reg", type=float, default=1e-1, help="L2 regularization on offsets (m^2).")
    p.add_argument("--stride", type=int, default=1, help="Use every k-th frame during fitting (1 = whole dataset).")
    p.add_argument("--method", choices=["Powell","Nelder-Mead"], default="Powell")
    p.add_argument("--maxiter", type=int, default=200)
    # Viz
    p.add_argument("--viz", action="store_true", help="Open Gepetto viewer (optional).")
    p.add_argument("--viz-sleep", type=float, default=0.02, help="Seconds between frames in viz loop.")
    return p.parse_args()

# def main():
args = parse_args()
base = Path(args.base_path)
mocap_csv = Path(args.mocap_csv) if args.mocap_csv else base / f"output/mocap/mocap_{args.subject_name}/{args.task}/mocap_downsampled_to_40hz.csv"
if not mocap_csv.exists():
    raise FileNotFoundError(mocap_csv)

# Load mocap + derive sequences (meters)
df = pd.read_csv(mocap_csv)
# df = udp_csv_to_dataframe(path_to_csv, mks_names)
df.columns = [col.replace(f"{args.subject_name}:", "") for col in df.columns]
mks_dict, start_sample_dict = read_mks_data(df, start_sample=0)

jcp_per_frame = []
meas_m = []

# Models
warmed = loadModel(augmenterDir=args.augmenter_dir, augmenterModelName="LSTM", augmenter_model=args.augmenter_model)

for frame_id in range(len(mks_dict)):
    markers_frame = mks_dict[frame_id]
    jcp, seg_lengths,norms = compute_joint_centers_from_mks(markers_frame)
    jcp_row = np.stack([
    np.asarray(jcp.get(name, np.full(3, np.nan))).reshape(3,)  # handles (3,) or (3,1)
    for name in JCP_NAMES
], axis=0) 
    jcp_per_frame.append(jcp_row.reshape(-1)/1000)       # (T, 60)
    meas_m.append( np.array([markers_frame[m] for m in MOCAP_MARKERS if m in markers_frame])/1000 )  

jcp_m_full = np.array(jcp_per_frame)

# Calculating augmented markers from base jcps
augmented_markers_list=[]
keypoints_buffer.clear()  # important: reset buffer for each eval

for ii in range(jcp_m_full.shape[0]):
    frame_data = jcp_m_full[ii,:].reshape(20, 3)
    if ii==0:
        for _ in range(30):
            keypoints_buffer.append(np.array(frame_data))
    else:
        keypoints_buffer.append(np.array(frame_data))
    
    if len(keypoints_buffer) == 30:
        keypoints_buffer_array = np.array(keypoints_buffer)
        augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=args.mass, subject_height=args.height, models = warmed,
                            augmenterDir=args.augmenter_dir, augmenter_model='v0.3')

        # ====== FILTER UNWANTED MARKERS HERE ======
        # augmented_markers is flat (129 = 43*3). Remove rows for thigh2/3 & sh2/3.
        aug = np.asarray(augmented_markers, dtype=float).reshape(43, 3)
        if DROP_IDX:  # delete those marker rows
            aug = np.delete(aug, DROP_IDX, axis=0)  # shape -> (43 - len(DROP_IDX), 3)
        augmented_markers = aug.reshape(-1)  # back to flat
        # ==========================================
        
        augmented_markers_list.append(augmented_markers)

augmented_array_base = np.vstack(augmented_markers_list)

meas_m = np.array(meas_m)

# Initial theta
theta0 = np.zeros((len(JCP_NAMES),3), dtype=float).ravel()

def apply_per_jcp_local_offsets(jcp_m_in, segments_frames_list, theta):
    """
    Apply per-JCP local offsets by rotating each local offset into world
    and adding it to the global JCP (no world->local->world needed).

    Args:
        jcp_m_in: (T_all, 60) global JCPs in meters.
        segments_frames_list: list of dicts, len = T_used. Each dict maps
            segment name -> 4x4 world_T_segment.
        theta: (60,) local offsets concatenated as (20*3,).

    Returns:
        (T_used, 60) global JCPs after applying offsets.
    """
    # Align to the frames we actually have segment poses for
    off = np.asarray(theta, dtype=float).reshape(len(JCP_NAMES), 3)

    def pick_seg(name, segs):
        mapping = {
            'RShoulder': 'torso',
            'LShoulder': 'torso',
            'Neck'     : 'torso',
            'RElbow'   : 'upperarmR',
            'LElbow'   : 'upperarmL',
            'RWrist'   : 'lowerarmR',
            'LWrist'   : 'lowerarmL',
            'RHip'     : 'pelvis', 
            'LHip'     : 'pelvis', 
            'midHip'   : 'pelvis', 
            'RKnee'    : 'thighR',
            'LKnee'    : 'thighL',
            'RAnkle'   : 'shankR',
            'LAnkle'   : 'shankL',
            'RHeel'    : 'footR',
            'LHeel'    : 'footL',
            'RBigToe'  : 'footR',
            'LBigToe'  : 'footL',
            'RSmallToe': 'footR',
            'LSmallToe': 'footL',
        }
        chosen = mapping.get(name)
        return chosen

    out = np.empty_like(jcp_m_in)
    for t, segs in enumerate(segments_frames_list):
        row = []
        for j_idx, jname in enumerate(JCP_NAMES):
            p = jcp_m_in[t, 3*j_idx:3*j_idx+3]
            if not np.all(np.isfinite(p)):
                row.append(p); continue
            seg = pick_seg(jname, segs)
            R_ws = segs[seg][:3, :3]
            d_world = R_ws @ off[j_idx]          # rotate local offset into world
            row.append(p + d_world)              # add directly to global JCP
        out[t, :] = np.concatenate(row)
    return out


# ---- Objective over WHOLE dataset (flattened vectors) ----
def objective(theta):
    markers_lstm=[]
    keypoints_buffer.clear()  # important: reset buffer for each eval

    for ii in range(jcp_m_full.shape[0]):
        frame_data = jcp_m_full[ii,:].reshape(20, 3)
        if ii==0:
            for _ in range(30):
                keypoints_buffer.append(np.array(frame_data))
        else:
            keypoints_buffer.append(np.array(frame_data))
        
        if len(keypoints_buffer) == 30:
            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=args.mass, subject_height=args.height, models = warmed,
                                augmenterDir=args.augmenter_dir, augmenter_model='v0.3')
            
            # ====== FILTER UNWANTED MARKERS HERE ======
            # augmented_markers is flat (129 = 43*3). Remove rows for thigh2/3 & sh2/3.
            aug = np.asarray(augmented_markers, dtype=float).reshape(43, 3)
            if DROP_IDX:  # delete those marker rows
                aug = np.delete(aug, DROP_IDX, axis=0)  # shape -> (43 - len(DROP_IDX), 3)
            augmented_markers = aug.reshape(-1)  # back to flat
            # ==========================================

            markers_lstm.append(dict(zip(MOCAP_MARKERS, np.reshape(augmented_markers, (-1, 3)))))

    segments_frames_list = []
    for jj in range(len(markers_lstm)):
        segments_frames = construct_segments_frames(markers_lstm[jj], with_hand=True, with_head=False)
        segments_frames_list.append(segments_frames)

    jcp_off = apply_per_jcp_local_offsets(jcp_m_full, segments_frames_list, theta)   # (T, 60)

    augmented_markers_list=[]
    keypoints_buffer.clear()  # important: reset buffer for each eval

    for ii in range(jcp_off.shape[0]):
        frame_data = jcp_off[ii,:].reshape(20, 3)
        if ii==0:
            for _ in range(30):
                keypoints_buffer.append(np.array(frame_data))
        else:
            keypoints_buffer.append(np.array(frame_data))
        
        if len(keypoints_buffer) == 30:
            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=args.mass, subject_height=args.height, models = warmed,
                                augmenterDir=args.augmenter_dir, augmenter_model='v0.3')
            
            # ====== FILTER UNWANTED MARKERS HERE ======
            # augmented_markers is flat (129 = 43*3). Remove rows for thigh2/3 & sh2/3.
            aug = np.asarray(augmented_markers, dtype=float).reshape(43, 3)
            if DROP_IDX:  # delete those marker rows
                aug = np.delete(aug, DROP_IDX, axis=0)  # shape -> (43 - len(DROP_IDX), 3)
            augmented_markers = aug.reshape(-1)  # back to flat
            # ==========================================

            augmented_markers_list.append(augmented_markers)

    augmented_array = np.vstack(augmented_markers_list)

    # print(augmented_array,meas_m.reshape(augmented_array.shape))
    val = np.sqrt(np.mean((meas_m.reshape(augmented_array.shape)-augmented_array)**2))

    if args.lambda_reg > 0:
        val += args.lambda_reg * float(np.dot(theta, theta))
    return val

res = minimize(objective, theta0, method=args.method, options=dict(maxiter=args.maxiter, disp=True))

# Full-resolution re-run with best theta (no stride)
best_theta = res.x
# best_theta, _ ,_  = load_offsets_json("/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/static/optimized_offset.json")

markers_lstm=[]
keypoints_buffer.clear()  # important: reset buffer for each eval

for ii in range(jcp_m_full.shape[0]):
    frame_data = jcp_m_full[ii,:].reshape(20, 3)
    if ii==0:
        for _ in range(30):
            keypoints_buffer.append(np.array(frame_data))
    else:
        keypoints_buffer.append(np.array(frame_data))
    
    if len(keypoints_buffer) == 30:
        keypoints_buffer_array = np.array(keypoints_buffer)
        augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=args.mass, subject_height=args.height, models = warmed,
                            augmenterDir=args.augmenter_dir, augmenter_model='v0.3')
        
        # ====== FILTER UNWANTED MARKERS HERE ======
        # augmented_markers is flat (129 = 43*3). Remove rows for thigh2/3 & sh2/3.
        aug = np.asarray(augmented_markers, dtype=float).reshape(43, 3)
        if DROP_IDX:  # delete those marker rows
            aug = np.delete(aug, DROP_IDX, axis=0)  # shape -> (43 - len(DROP_IDX), 3)
        augmented_markers = aug.reshape(-1)  # back to flat
        # ==========================================

        markers_lstm.append(dict(zip(MOCAP_MARKERS, np.reshape(augmented_markers, (-1, 3)))))

segments_frames_list = []
for jj in range(len(markers_lstm)):
    segments_frames = construct_segments_frames(markers_lstm[jj], with_hand=True, with_head=False)
    segments_frames_list.append(segments_frames)

jcp_off_full = apply_per_jcp_local_offsets(jcp_m_full, segments_frames_list, best_theta)
# jcp_off_full = apply_per_jcp_local_offsets(jcp_m_full, segments_frames_list, np.array(list(best_theta.values())))
                                                    # (T, M_all, 3)

augmented_markers_list=[]
keypoints_buffer.clear()  # important: reset buffer for each eval

for ii in range(jcp_off_full.shape[0]):
    frame_data = jcp_off_full[ii,:].reshape(20, 3)
    if ii==0:
        for _ in range(30):
            keypoints_buffer.append(np.array(frame_data))
    else:
        keypoints_buffer.append(np.array(frame_data))
    
    if len(keypoints_buffer) == 30:
        keypoints_buffer_array = np.array(keypoints_buffer)
        augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=args.mass, subject_height=args.height, models = warmed,
                            augmenterDir=args.augmenter_dir, augmenter_model='v0.3')

        # ====== FILTER UNWANTED MARKERS HERE ======
        # augmented_markers is flat (129 = 43*3). Remove rows for thigh2/3 & sh2/3.
        aug = np.asarray(augmented_markers, dtype=float).reshape(43, 3)
        if DROP_IDX:  # delete those marker rows
            aug = np.delete(aug, DROP_IDX, axis=0)  # shape -> (43 - len(DROP_IDX), 3)
        augmented_markers = aug.reshape(-1)  # back to flat
        # ==========================================
        
        augmented_markers_list.append(augmented_markers)

augmented_array = np.vstack(augmented_markers_list)

rmse = np.sqrt(np.mean((meas_m.reshape(augmented_array.shape)-augmented_array)**2))
rmse_base = np.sqrt(np.mean((meas_m.reshape(augmented_array.shape)-augmented_array_base)**2))

print(f"[Per-JCP] Best global RMSE with scipy f={res.fun:.6f} m")
print(f"[Per-JCP] Best global RMSE with rmse f={rmse} m")
print(f"[Per-JCP] RMSE with rmse_base f={rmse_base} m")

# Save CSV (meters)
out_csv = Path(args.out) if args.out else (base / f"output/{args.subject_id}/cosmik_2cams/{args.task}/augmented_markers_perjcp_optimized.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
save_to_csv(augmented_array.reshape(augmented_array.shape[0], -1), out_csv, header=MOCAP_HEADER)
print(f"[OK] Saved optimized augmented markers to: {out_csv}")

# Save offsets JSON (meters)
off_json = Path(args.save_offsets_json) if args.save_offsets_json else out_csv.with_suffix(".offsets.json")
per = best_theta.reshape(len(JCP_NAMES),3)
payload = {"mode": "per-jcp", "offsets_per_jcp_m": {JCP_NAMES[i]: per[i].tolist() for i in range(len(JCP_NAMES))}}
with open(off_json, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[OK] Saved offsets to: {off_json}")

# (Optional) Viz
if args.viz:
    input("Press Enter to start visualization...")
    visualize_simple(
        preds_full=augmented_array,
        meas_m_full=meas_m.reshape(augmented_array.shape),
        aug_markers_base=augmented_array_base,
        jcp_m_full=jcp_m_full,
        jcp_off_full=jcp_off_full,
        sleep_dt=args.viz_sleep
    )


