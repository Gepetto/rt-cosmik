import os
import sys
import numpy as np
import pandas as pd

mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

mks_names_extended = ["Frame", "frame"]
for mks in mks_names:
    mks_names_extended.append(f"{mks}_x")
    mks_names_extended.append(f"{mks}_y")
    mks_names_extended.append(f"{mks}_z")

dataset_path = sys.argv[1]

for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)
    if subject != "Mathis":
        for trial in os.listdir(subject_path):
            trial_path = os.path.join(subject_path, trial)
            mks_data_rt_path = os.path.join(trial_path, "mks_data_gapfilled.csv")
            out_npz_path = os.path.join(trial_path, "mks_data_gapfilled.npz")
            df_mks_mocap = pd.read_csv(mks_data_rt_path, names=mks_names_extended)
            df_mks_mocap = df_mks_mocap.iloc[:, 2:]
            data = df_mks_mocap.to_numpy(dtype=np.float32, copy=False)
            cols = df_mks_mocap.columns.to_numpy()
            np.savez_compressed(out_npz_path, data=data, columns=cols)