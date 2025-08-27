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
from src.rtcosmik.augmenter.marker_augmenter import augmentTRC, loadModel, augmentTRCOpenCap, loadModelOpenCap
from src.rtcosmik.utils.read_write_utils import read_mmpose_file, save_to_csv, read_subject_info
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter

base_path = "/pfcalcul/work/ngouget/"

p = argparse.ArgumentParser(description="augment data w local lstm")
p.add_argument('--use-mocap', choices=['T','F'], default='T', required=True)        # must be 'T' for this script (mocap JCP + mocap GT)
p.add_argument('--add-noise', choices=['T','F'], default='F')
p.add_argument('--fine-tune', choices=['T','F'], default='F')
p.add_argument('--add-layer', choices=['T','F'], default='F')
p.add_argument('--use-weights', choices=['T','F'], default='F')
p.add_argument('--rot-prob', type=float, default=0.0, help='Probability to apply a random yaw rotation per window (0..1)')
p.add_argument('--rot-max-deg', type=float, default=30.0, help='Max absolute rotation in degrees (uniform in [-max, max])')
p.add_argument('--up-axis', choices=['y','z'], default='z', help='Which axis is vertical in your data (usually y or z)')
p.add_argument('--rotation-scheme', choices=['off','prob','det'], default='off',
               help="off: no rotation; prob: single random yaw per window using --rot-prob/--rot-max-deg; det: emit n evenly-spaced yaws per window")
p.add_argument('--n-rotations', type=int, default=1,
               help="When --rotation-scheme det, emit this many evenly-spaced yaw angles per window (full circle).")
p.add_argument('--subject', type=str, default=None)
p.add_argument('--trial', type=str, default=None)

args = p.parse_args()


subject_path = f"/pfcalcul/work/ngouget/COSMIK_dataset_raw/{args.subject}"
trial_path = os.path.join(subject_path, args.trial)
if args.use_mocap == "T":
    converter = 1000.0
    path_to_3d_kpt = os.path.join(trial_path, f"{args.trial}_jcp_mocap.csv")
    output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.csv")
    output_csv_path_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_mocap_OpenCap.csv")
elif args.use_mocap == "F":
    converter = 1.0
    path_to_3d_kpt = os.path.join(trial_path, f"{args.trial}_jcp_hpe.csv")
    output_csv_path = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_hpe_ft{args.fine_tune}_al{args.add_layer}_m{args.use_mocap}_n{args.add_noise}_w{args.use_weights}_prot{args.rot_prob}_maxrot{args.rot_max_deg}_rotscheme{args.rotation_scheme}_up{args.up_axis}_nrot{args.n_rotations}.csv")
    output_csv_path_OpenCap = os.path.join(base_path, f"rt-cosmik/output/{args.subject}/{args.trial}/{args.trial}_augmented_markers_hpe_OpenCap.csv")
else :
    raise Exception("Use mocap not supported. Please select T or F.")


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
    warmed_models = loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3', use_mocap=args.use_mocap, add_noise=args.add_noise, fine_tune=args.fine_tune, 
                                add_layer=args.add_layer, use_weights=args.use_weights, rot_prob=args.rot_prob, rot_max_deg=args.rot_max_deg, rotation_scheme=args.rotation_scheme, n_rotations=args.n_rotations)
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
            for _ in range(30):
                keypoints_buffer.append(np.array(frame_data/converter))
            first_frame = False 
        else:
            keypoints_buffer.append(np.array(frame_data/converter))

        if len(keypoints_buffer) == 30:

            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models,
                                augmenterDir=augmenter_path, augmenter_model='v0.3', use_mocap=args.use_mocap, add_noise=args.add_noise, fine_tune=args.fine_tune, 
                                add_layer=args.add_layer, use_weights=args.use_weights, rot_prob=args.rot_prob, rot_max_deg=args.rot_max_deg, rotation_scheme=args.rotation_scheme, n_rotations=args.n_rotations)
            augmented_markers_opencap = augmentTRCOpenCap(keypoints_buffer_array, subject_mass=subject_weight, subject_height=subject_height, models = warmed_models_opencap,
                                augmenterDir=augmenter_path, augmenter_model='v0.3', offset=False, use_mocap=args.use_mocap)
            augmented_markers_list.append(augmented_markers)
            augmented_markers_list_opencap.append(augmented_markers_opencap)

    augmented_array = np.vstack(augmented_markers_list) 
    augmented_array_opencap = np.vstack(augmented_markers_list_opencap)

    # filtered_data = butterworth_filter(
    # data=augmented_array,
    # cutoff_frequency=10.0,  
    # order=5,
    # sampling_frequency=40
    # )
    save_to_csv(augmented_array, output_csv_path, header=header)
    save_to_csv(augmented_array_opencap, output_csv_path_OpenCap, header=header)
    

if __name__ == "__main__":
    main()
