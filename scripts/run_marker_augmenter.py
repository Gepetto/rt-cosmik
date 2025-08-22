#run augmenter on csv file 
import os
import sys
import pandas as pd
from collections import deque
import numpy as np
from scipy import signal
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv, read_subject_info
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter

base_path = "/home/ngouget/Codes/"

subject = sys.argv[1]
trial = sys.argv[2]
use_mocap = "T"
add_noise = "F"
subject_path = os.path.join(base_path, f"datasets/COSMIK_dataset/{subject}")
trial_path = os.path.join(subject_path, trial)
path_to_3d_kpt = os.path.join(trial_path, f"{trial}_jcp_mocap.csv")
output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{subject}/{trial}/{trial}_augmented_markers_test.csv")

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
    first_frame = True
    #load lstm model
    warmed_models = loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

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
            for _ in range(30):
                keypoints_buffer.append(np.array(frame_data))
            first_frame = False 
        else:
            keypoints_buffer.append(np.array(frame_data))

        if len(keypoints_buffer) == 30:

            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models,
                                augmenterDir=augmenter_path, augmenter_model='v0.3', use_mocap=use_mocap, add_noise=add_noise)
            augmented_markers_list.append(augmented_markers)

    augmented_array = np.vstack(augmented_markers_list) 

    # filtered_data = butterworth_filter(
    # data=augmented_array,
    # cutoff_frequency=10.0,  
    # order=5,
    # sampling_frequency=40
    # )
    save_to_csv(augmented_array, output_csv_path, header=header)
    

if __name__ == "__main__":
    main()
