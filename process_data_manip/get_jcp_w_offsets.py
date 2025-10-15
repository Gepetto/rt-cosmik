#!/usr/bin/env python3
import os, json, argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.joinpath("../../../src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.rtcosmik.utils.read_write_utils import read_mks_data, save_to_csv, read_subject_info

# ---- rtcosmik pieces (same as your optimizer) ----
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from src.rtcosmik.human_model.model_utils import construct_segments_frames
from process_data_manip.get_jcp_from_anatomical_mks import compute_joint_centers_from_mks


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

# mapping JCP -> segment frame (same as optimizer)
SEG_FOR_JCP = {
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

# set used by the optimizer to drop unwanted augmented rows
TO_DROP = {
    'r_thigh1_study','r_thigh2_study','r_thigh3_study',
    'L_thigh1_study','L_thigh2_study','L_thigh3_study',
    'r_sh1_study','r_sh2_study','r_sh3_study',
    'L_sh1_study','L_sh2_study','L_sh3_study',
    'RHJC_study','LHJC_study'
}

# augmented markers full list (43)
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
DROP_IDX = [i for i, n in enumerate(AUGMENTED_MARKERS) if n in TO_DROP]

# comparison subset (same as optimizer’s MOCAP_MARKERS) & header helper
MOCAP_MARKERS = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
    'C7_study','r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
]
def jcp_header(names=JCP_NAMES):
    return [f"{n}_{ax}" for n in names for ax in ("x","y","z")]

# ==============
# Helper functions
# ==============
def load_df_from_npz(npz_path: str) -> pd.DataFrame:
    npz = np.load(npz_path, allow_pickle=True)
    columns = npz["columns"]
    data = npz["data"]
    if columns.ndim > 1:
        columns = columns.ravel()
    columns = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in columns]
    df = pd.DataFrame(data, columns=columns)
    if "Frame" not in df.columns:
        df.insert(0, "Frame", np.arange(len(df)))
    return df

def load_offsets_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    offs = data["offsets_per_jcp_m"]  # meters
    off_mat = np.stack([np.asarray(offs.get(name, [0.0, 0.0, 0.0]), dtype=float)
                        for name in JCP_NAMES], axis=0)  # (20,3)
    theta = off_mat.reshape(-1)  # (60,)
    return theta

def apply_per_jcp_local_offsets(jcp_m_in: np.ndarray, segments_frames_list, theta: np.ndarray) -> np.ndarray:
    """
    jcp_m_in: (T, 60) global JCPs (meters)
    segments_frames_list: len=T, each dict: segment -> 4x4 world_T_segment
    theta: (60,) local offsets per JCP (meters in segment frame)
    returns: (T, 60) global JCPs after offsets
    """
    off = np.asarray(theta, float).reshape(len(JCP_NAMES), 3)
    out = np.empty_like(jcp_m_in)
    for t, segs in enumerate(segments_frames_list):
        row = []
        for j_idx, jname in enumerate(JCP_NAMES):
            p = jcp_m_in[t, 3*j_idx:3*j_idx+3]
            if not np.all(np.isfinite(p)):
                row.append(p); continue
            seg = SEG_FOR_JCP[jname]
            R_ws = segs[seg][:3, :3]
            d_world = R_ws @ off[j_idx]
            row.append(p + d_world)
        out[t, :] = np.concatenate(row)
    return out

def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

# =========
# Main
# =========
def main():
    ap = argparse.ArgumentParser(description="NPZ → JCPs with per-JCP offsets (meters) + RMSE report.")
    ap.add_argument("--root", default="./mocap_100hz",
                    help="Folder containing {subject}/{task}/{task}_trajectories.npz")
    ap.add_argument("--subjects", nargs="*", default=[
        "Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie",
        "Guilhem","Kahina","Marie","Mathis","Maxime","Mohamed","Nicolas","Zoe","Herbert","Emmanuelle"
    ])
    ap.add_argument("--tasks", nargs="*", default=["robot_welding"])
    ap.add_argument("--npz-name", default="{task}_trajectories.npz")
    ap.add_argument("--out-root", default="./mocap_jcp_csv_with_offsets")
    # LSTM augmenter
    ap.add_argument("--augmenter-dir", default="src/rtcosmik/augmenter/augmentation_model")
    ap.add_argument("--augmenter-model", default="v0.3")

    # Offsets
    ap.add_argument("--offsets-json", required=True, help="JSON with offsets_per_jcp_m (meters)")
    # Units
    ap.add_argument("--input-units", choices=["m","mm"], default="mm",
                    help="Units of NPZ marker data expected by compute_joint_centers_from_mks")
    args = ap.parse_args()

    # load offsets
    theta = load_offsets_json(args.offsets_json)  # (60,)

    # load augmenter
    warmed = loadModel(augmenterDir=args.augmenter_dir,
                       augmenterModelName="LSTM",
                       augmenter_model=args.augmenter_model)

    mm_to_m = 0.001 if args.input_units == "mm" else 1.0

    for subject in args.subjects:
        info_path =f"/root/workspace/ros_ws/src/rt-cosmik/output/mocap_100hz/{subject}/info.txt"
        height, mass, gender = read_subject_info(str(info_path))
        for task in args.tasks:
            base_dir = os.path.join(args.root, subject, task)
            # resolve NPZ path
            if "{task}" in args.npz_name:
                npz_path = os.path.join(base_dir, args.npz_name.format(task=task))
            else:
                npz_path = os.path.join(base_dir, args.npz_name)

            print(f"\n[{subject}/{task}] Loading NPZ: {npz_path}")
            if not os.path.exists(npz_path):
                print(f"  -> Skipping (not found)")
                continue

            df = load_df_from_npz(npz_path)
            # deprefix columns "Subject:"
            df.columns = [c.replace(f"{subject}:", "") for c in df.columns]

            # markers dict per frame (keep raw units; we convert JCP later)
            mks_dict, _ = read_mks_data(df, start_sample=0, converter=1000.0)

            # 1) base JCPs (global) — in input units; then convert to meters
            jcp_rows = []
            for t in range(len(mks_dict)):
                # UNPACK the tuple your function returns:
                jcp_global, _, _ = compute_joint_centers_from_mks(mks_dict[t])  # dict[str]->(3,)
                row = []
                for name in JCP_NAMES:
                    coords = jcp_global.get(name, [np.nan, np.nan, np.nan])
                    p = np.asarray(coords, float).reshape(3,)
                    row.extend(p.tolist())
                jcp_rows.append(row)
            jcp_arr = np.asarray(jcp_rows, float)          # (T,60) input units
            jcp_m_full = jcp_arr                  # meters

            # 2) Build segment frames from baseline LSTM augmented markers
            keypoints_buffer = deque(maxlen=30)
            markers_lstm = []  # each: dict(name->(3,))
            for t in range(jcp_m_full.shape[0]):
                frame_jcp = jcp_m_full[t, :].reshape(len(JCP_NAMES), 3)
                if t == 0:
                    for _ in range(30):
                        keypoints_buffer.append(frame_jcp.copy())
                else:
                    keypoints_buffer.append(frame_jcp.copy())
                if len(keypoints_buffer) == 30:
                    hist = np.asarray(keypoints_buffer)  # (30,20,3)
                    aug = augmentTRC(
                        hist,
                        subject_mass=mass,
                        subject_height=height,
                        models=warmed,
                        augmenterDir=args.augmenter_dir,
                        augmenter_model=args.augmenter_model
                    )
                    aug = np.asarray(aug, float).reshape(len(AUGMENTED_MARKERS), 3)
                    if DROP_IDX:
                        aug = np.delete(aug, DROP_IDX, axis=0)
                    d = dict(zip(MOCAP_MARKERS, aug.reshape(-1,3)))
                    markers_lstm.append(d)

            # 3) segment frames from baseline markers
            segments_frames_list = [construct_segments_frames(d, with_hand=True, with_head=False, gender=gender)
                                    for d in markers_lstm]

            # Align lengths (first 29 frames have no LSTM output)
            jcp_m_used = jcp_m_full  # (T-29, 60)

            # 4) apply per-JCP local offsets (meters, segment frames)
            jcp_off_used = apply_per_jcp_local_offsets(jcp_m_used, segments_frames_list, theta)  # (T-29,60)

            # 5) Re-augment from offset JCPs (for RMSE_off)
            keypoints_buffer.clear()
            aug_off_list = []
            for t in range(jcp_off_used.shape[0]):
                frame_jcp = jcp_off_used[t, :].reshape(len(JCP_NAMES), 3)
                if t == 0:
                    for _ in range(30):
                        keypoints_buffer.append(frame_jcp.copy())
                else:
                    keypoints_buffer.append(frame_jcp.copy())
                if len(keypoints_buffer) == 30:
                    
                    hist = np.asarray(keypoints_buffer)
                    aug = augmentTRC(
                        hist,
                        subject_mass=mass,
                        subject_height=height,
                        models=warmed,
                        augmenterDir=args.augmenter_dir,
                        augmenter_model=args.augmenter_model
                    )
                    aug = np.asarray(aug, float).reshape(len(AUGMENTED_MARKERS), 3)
                    if DROP_IDX:
                        aug = np.delete(aug, DROP_IDX, axis=0)
                    d = dict(zip(MOCAP_MARKERS, aug.reshape(-1,3)))
                    row = []
                    for name in MOCAP_MARKERS:
                        row.extend(np.asarray(d[name], float).reshape(3,))
                    aug_off_list.append(row)
            # second 30-frame history → shorter again
            aug_off = np.asarray(aug_off_list, float)  # (T-29-29, M*3) = (T-58, M*3)

            # Baseline augmented (already built once from markers_lstm)
            aug_base_list = []
            for d in markers_lstm:
                row = []
                for name in MOCAP_MARKERS:
                    row.extend(np.asarray(d[name], float).reshape(3,))
                aug_base_list.append(row)
            aug_base_full = np.asarray(aug_base_list, float)  # (T-29, M*3)

            # measured markers array in meters (for RMSE)
            meas_rows = []
            for t in range(len(mks_dict)):
                mf = mks_dict[t]
                row = []
                for name in MOCAP_MARKERS:
                    if name in mf:
                        row.extend(np.asarray(mf[name], float).reshape(3,))
                    else:
                        row.extend([np.nan, np.nan, np.nan])
                meas_rows.append(row)
            meas_m_full = np.asarray(meas_rows, float)  # (T, M*3)

            # For RMSE we need alignment on tightest window (T-58)
            n_base = aug_base_full.shape[0]          # T-29
            n_off  = aug_off.shape[0]                # T-58
            if n_off <= 0:
                print("  -> Not enough frames for offset RMSE. Will still save JCP CSV.")
                rmse_base_val = np.nan
                rmse_off_val = np.nan
            else:
                meas_for_base = meas_m_full[-n_base:, :]
                meas_for_off  = meas_m_full[-n_off:, :]
                base_for_off  = aug_base_full[-n_off:, :]  # compare same window size
                rmse_base_val = rmse(meas_m_full, aug_base_full)
                rmse_off_val  = rmse(meas_m_full, aug_off)
                print(f"RMSE baseline (no offsets): {rmse_base_val:.6f} m")
                print(f"RMSE with offsets:         {rmse_off_val:.6f} m")

            # 6) Build full-length JCP with offsets for CSV:
            # pad first 29 frames with NaNs; jcp_off_used covers T-29 frames.
            jcp_off_full = jcp_off_used   # (T,60)

            out_dir = os.path.join(args.out_root, subject, task)
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, "joint_center_positions_with_offsets.csv")
            pd.DataFrame(jcp_off_full, columns=jcp_header(JCP_NAMES)).to_csv(out_csv, index=False)
            print(f"  Saved JCPs with offsets (meters): {out_csv}")

            # optional: also write RMSEs to a tiny sidecar JSON for provenance
            rmse_json = os.path.join(out_dir, "rmse_summary.json")
            with open(rmse_json, "w") as f:
                json.dump({
                    "rmse_baseline_m": None if not np.isfinite(rmse_base_val) else float(rmse_base_val),
                    "rmse_with_offsets_m": None if not np.isfinite(rmse_off_val) else float(rmse_off_val),
                    "notes": "RMSE computed on tightest common window; first 29 or 58 frames excluded due to LSTM history."
                }, f, indent=2)
            print(f"  Saved RMSE summary: {rmse_json}")

if __name__ == "__main__":
    main()
