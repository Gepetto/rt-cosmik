#run augmenter on csv file 
import os
import sys
import pandas as pd
from collections import deque
import numpy as np
from scipy import signal
from pathlib import Path
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel, augmentTRCOpenCap, loadModelOpenCap, loadModel_simple, augmentTRC_simple
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv, read_subject_info
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter

base_path = "/home/ngouget/Codes"
augmenter_path = os.path.join(base_path, 'rt-cosmik/src/rtcosmik/augmenter/augmentation_model')

order_jcp_must_have = ['RShoulder_x', 'RShoulder_y', 'RShoulder_z', 
                        'LShoulder_x', 'LShoulder_y', 'LShoulder_z', 
                        'Neck_x', 'Neck_y', 'Neck_z', 'RElbow_x', 
                        'RElbow_y', 'RElbow_z', 'LElbow_x', 'LElbow_y', 
                        'LElbow_z', 'RWrist_x', 'RWrist_y', 'RWrist_z', 
                        'LWrist_x', 'LWrist_y', 'LWrist_z', 'RHip_x', 
                        'RHip_y', 'RHip_z', 'LHip_x', 'LHip_y', 'LHip_z', 
                        'midHip_x', 'midHip_y', 'midHip_z', 'RKnee_x', 
                        'RKnee_y', 'RKnee_z', 'LKnee_x', 'LKnee_y', 'LKnee_z', 
                        'RAnkle_x', 'RAnkle_y', 'RAnkle_z', 'LAnkle_x', 'LAnkle_y', 
                        'LAnkle_z', 'RHeel_x', 'RHeel_y', 'RHeel_z', 'LHeel_x', 'LHeel_y', 
                        'LHeel_z', 'RBigToe_x', 'RBigToe_y', 'RBigToe_z', 'LBigToe_x', 
                        'LBigToe_y', 'LBigToe_z', 'RSmallToe_x', 'RSmallToe_y', 'RSmallToe_z', 
                        'LSmallToe_x', 'LSmallToe_y', 'LSmallToe_z']

p = argparse.ArgumentParser(description="augment data w local lstm")
p.add_argument('--test-over-mocap', choices=['T','F'], default='T', required=True)        # must be 'T' for this script (mocap JCP + mocap GT)
p.add_argument('--upper-model-path', type=str, required=True)
p.add_argument('--lower-model-path', type=str, required=True)

args = p.parse_args()

model_name_upper = os.path.basename(args.upper_model_path)[:-5]
model_name_lower = os.path.basename(args.lower_model_path)[:-5]

old_config = False
last_char = "a"
for ind, char in enumerate(model_name_upper):
    current_char = char
    if last_char == "f" and current_char == "t":
        old_config = True
        ind_ft = ind - 1
        break
    last_char = current_char

if old_config:
    opt_name_upper = model_name_upper[ind_ft:]
    opt_name_lower = model_name_lower[ind_ft:]
else:
    opt_name_upper = "final_" + model_name_upper[26:]
    opt_name_lower = "final_" + model_name_lower[26:]

included_trials = ["robot_welding"]
included_subjects = ["Kahina", "Flavie"]

procrustes = True

def kabsch_global(P_cam_seq, P_mocap_seq, weights=None):
    """
    P_cam_seq, P_mocap_seq: arrays (T, N, 3) alignés temporellement et par point.
    Calcule UN seul (R,t) qui aligne tout (cam -> mocap) en minimisant la somme des erreurs.
    """

    assert P_cam_seq.shape == P_mocap_seq.shape and P_cam_seq.shape[-1] == 3
    T, N, _ = P_cam_seq.shape
    X = P_cam_seq.reshape(T*N, 3)
    Y = P_mocap_seq.reshape(T*N, 3)

    if weights is not None:
        w = np.asarray(weights).reshape(T, N)
        w = w / (w.sum() + 1e-12)
        w = w.reshape(T*N, 1)
        Xc = (X * w).sum(axis=0)     # weighted means
        Yc = (Y * w).sum(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = (Y0 * w).T @ X0
    else:
        Xc = X.mean(axis=0)
        Yc = Y.mean(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = Y0.T @ X0

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # corrige réflexion
        U[:, -1] *= -1
        R = U @ Vt
    t = Yc - R @ Xc

    X_align = (R @ X.T).T + t
    rms = np.sqrt(np.mean(np.sum((X_align - Y)**2, axis=1)))
    return R, t, rms

def apply_transform_df(df, R, t):
    """
    Applique R, t à un DataFrame (T, 3N) de colonnes x,y,z concaténées.
    """
    arr = df.to_numpy()                # (T,3N)
    T, C = arr.shape
    assert C % 3 == 0
    N = C // 3

    # reshape en (T,N,3), appliquer la transfo, re-flatten
    arr3 = arr.reshape(T, N, 3)
    arr3_aligned = (arr3 @ R.T) + t
    arr_aligned = arr3_aligned.reshape(T, C)

    return pd.DataFrame(arr_aligned, columns=df.columns, index=df.index)




markers = [
           'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study',
           'r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

header = []
for marker in markers:
    header.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

keypoints_buffer = deque(maxlen=30)


def main():

    for subject in included_subjects:
        subject_path = f"/home/ngouget/Codes/datasets/COSMIK_dataset_mixed/{subject}"
        info_path = Path(os.path.join(subject_path, "info.txt"))
        subject_height, subject_weight , _ = read_subject_info(info_path)
        for trial in included_trials:
            trial_path = os.path.join(subject_path, trial)
            if args.test_over_mocap == "T":
                converter = 1.0
                path_to_3d_kpt = os.path.join(trial_path, f"{trial}_jcp_mocap.csv")
                output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{subject}/{trial}/{trial}_augmented_markers_mocap_{model_name_lower}.csv")
                output_csv_path_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{subject}/{trial}/{trial}_augmented_markers_mocap_OpenCap.csv")
            elif args.test_over_mocap == "F":
                converter = 1.0
                path_to_3d_kpt = os.path.join(trial_path, f"{trial}_jcp_hpe.csv")
                output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{subject}/{trial}/{trial}_augmented_markers_hpe_{model_name_lower}.csv")
                output_csv_path_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{subject}/{trial}/{trial}_augmented_markers_hpe_OpenCap.csv")
            else :
                raise Exception("Use mocap not supported. Please select T or F.")

            if procrustes:
                path_to_jcp_mocap = os.path.join(trial_path, f"{trial}_jcp_mocap.csv")
                data_jcp_hpe = pd.read_csv(path_to_3d_kpt)
                data_jcp_hpe = data_jcp_hpe[order_jcp_must_have]
                data_jcp_mocap = pd.read_csv(path_to_jcp_mocap)
                jcp_mocap = (np.array(data_jcp_mocap.values)).reshape(data_jcp_mocap.shape[0], 20, 3)
                jcp_hpe = (np.array(data_jcp_hpe.values)).reshape(data_jcp_hpe.shape[0], 20, 3)
                R, t, rms_error = kabsch_global(jcp_hpe, jcp_mocap)

                data_jcp_hpe = apply_transform_df(data_jcp_hpe, R, t)

            augmented_markers_list = []
            augmented_markers_list_opencap = []
            first_frame = True
            #load lstm model
            warmed_models = loadModel_simple(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3', model_name_upper=model_name_upper, model_name_lower=model_name_lower)
            # if not os.path.exists(output_csv_path):
            warmed_models_opencap = loadModelOpenCap(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

            #load 3d keypoints
            if procrustes:
                data_jcp_hpe.to_csv(f"{path_to_3d_kpt[:-4]}_aligned.csv", index=False)
                data = data_jcp_hpe.values
            else:
                data = pd.read_csv(path_to_3d_kpt).values
            num_columns = data.shape[1]

            if num_columns % 3 != 0:
                raise ValueError(f"Unexpected number of columns: {num_columns}. It should be divisible by 3.") #60/3 = 20

            num_keypoints = num_columns // 3
            coordinates_per_keypoint = 3
            
            #apply the lstm on the data
            for i in range(len(data)):
                #reshape each frame into a (num_keypoints, 3) array and add to the buffer
                frame_data = data[i].reshape(num_keypoints, coordinates_per_keypoint)

                if first_frame:
                    for _ in range(30):
                        keypoints_buffer.append(np.array(frame_data/converter))
                    first_frame = False 
                else:
                    keypoints_buffer.append(np.array(frame_data/converter))

                if len(keypoints_buffer) == 30:

                    keypoints_buffer_array = np.array(keypoints_buffer)
                    augmented_markers = augmentTRC_simple(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models,
                                        augmenterDir=augmenter_path, augmenter_model='v0.3', opt_name_upper=opt_name_upper, opt_name_lower=opt_name_lower)
                    augmented_markers_list.append(augmented_markers)
                    # if not os.path.exists(output_csv_path_OpenCap):
                    augmented_markers_opencap = augmentTRCOpenCap(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models_opencap,
                                        augmenterDir=augmenter_path, augmenter_model='v0.3', offset=False)
                    augmented_markers_list_opencap.append(augmented_markers_opencap)            

            augmented_array = np.vstack(augmented_markers_list) 
            save_to_csv(augmented_array, output_csv_path, header=header)

            # if not os.path.exists(output_csv_path_OpenCap):
            augmented_array_opencap = np.vstack(augmented_markers_list_opencap)
            save_to_csv(augmented_array_opencap, output_csv_path_OpenCap, header=header)

            # filtered_data = butterworth_filter(
            # data=augmented_array,
            # cutoff_frequency=10.0,  
            # order=5,
            # sampling_frequency=40
            # )
    

if __name__ == "__main__":
    main()
