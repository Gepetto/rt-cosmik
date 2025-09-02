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
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel, augmentTRCOpenCap, loadModelOpenCap, loadModel_incHPE, augmentTRC_incHPE
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv, read_subject_info
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter

base_path = "/home/ngouget/Codes"

p = argparse.ArgumentParser(description="augment data w local lstm")
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--seq-len', type=int, default=30)
p.add_argument("--exclude-trials", type=str, default="none", help="Exclude trials from the dataset")
p.add_argument('--subject', type=str, default=None)
p.add_argument('--trial', type=str, default=None)

args = p.parse_args()


subject_path = f"/home/ngouget/Codes/datasets/COSMIK_dataset_mixed/{args.subject}"
trial_path = os.path.join(subject_path, args.trial)

path_to_3d_kpt = os.path.join(trial_path, f"{args.trial}_jcp_hpe.csv")
output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_n{args.add_noise}_w{args.use_weights}_sl{args.seq_len}_exclude{args.exclude_trials}.csv")
output_csv_path_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_OpenCap.csv")

info_path = Path(os.path.join(subject_path, "info.txt"))
subject_height, subject_weight , _ = read_subject_info(info_path)
augmenter_path = os.path.join(base_path, 'rt-cosmik/src/rtcosmik/augmenter/augmentation_model')
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
    augmented_markers_list = []
    augmented_markers_list_opencap = []
    first_frame = True
    #load lstm model
    warmed_models = loadModel_incHPE(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3', add_noise=args.add_noise,
                             use_weights=args.use_weights, seq_len=args.seq_len, exclude_trials=args.exclude_trials)
    if not os.path.exists(output_csv_path):
        warmed_models_opencap = loadModelOpenCap(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

    #load 3d keypoints
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
            for _ in range(args.seq_len):
                keypoints_buffer.append(np.array(frame_data))
            first_frame = False 
        else:
            keypoints_buffer.append(np.array(frame_data))

        if len(keypoints_buffer) == args.seq_len:

            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC_incHPE(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models,
                                augmenterDir=augmenter_path, augmenter_model='v0.3', add_noise=args.add_noise,
                                use_weights=args.use_weights, seq_len=args.seq_len, exclude_trials=args.exclude_trials)
            augmented_markers_list.append(augmented_markers)
            if not os.path.exists(output_csv_path_OpenCap):
                augmented_markers_opencap = augmentTRCOpenCap(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models_opencap,
                                    augmenterDir=augmenter_path, augmenter_model='v0.3', offset=False)
                augmented_markers_list_opencap.append(augmented_markers_opencap)            

    augmented_array = np.vstack(augmented_markers_list) 
    save_to_csv(augmented_array, output_csv_path, header=header)

    if not os.path.exists(output_csv_path_OpenCap):
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
