import os
import sys
import pandas as pd

data_path = sys.argv[1]

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


if os.path.basename(data_path) == "mks_data_gapfilled.csv":
    df = pd.read_csv(data_path, names=mks_names_extended)
    df = df.iloc[:, 2:]

    df.index.name = "Frame"

    order_mks_must_have = ['r.PSIS_study_x', 'r.PSIS_study_y', 'r.PSIS_study_z', 
                        'L.PSIS_study_x', 'L.PSIS_study_y', 'L.PSIS_study_z', 'r.ASIS_study_x', 
                        'r.ASIS_study_y', 'r.ASIS_study_z', 'L.ASIS_study_x', 'L.ASIS_study_y', 
                        'L.ASIS_study_z', 'r_pelvis_x', 'r_pelvis_y', 'r_pelvis_z', 'l_pelvis_x', 
                        'l_pelvis_y', 'l_pelvis_z', 'r_knee_study_x', 'r_knee_study_y', 'r_knee_study_z', 
                        'r_mknee_study_x', 'r_mknee_study_y', 'r_mknee_study_z', 'r_thigh1_study_x', 
                        'r_thigh1_study_y', 'r_thigh1_study_z', 'r_ankle_study_x', 'r_ankle_study_y', 
                        'r_ankle_study_z', 'r_mankle_study_x', 'r_mankle_study_y', 'r_mankle_study_z', 
                        'r_sh1_study_x', 'r_sh1_study_y', 'r_sh1_study_z', 'r_calc_study_x', 'r_calc_study_y', 
                        'r_calc_study_z', 'r_5meta_study_x', 'r_5meta_study_y', 'r_5meta_study_z', 'r_toe_study_x', 
                        'r_toe_study_y', 'r_toe_study_z', 'L_knee_study_x', 'L_knee_study_y', 'L_knee_study_z', 
                        'L_mknee_study_x', 'L_mknee_study_y', 'L_mknee_study_z', 'L_thigh1_study_x', 'L_thigh1_study_y', 
                        'L_thigh1_study_z', 'L_ankle_study_x', 'L_ankle_study_y', 'L_ankle_study_z', 'L_mankle_study_x', 
                        'L_mankle_study_y', 'L_mankle_study_z', 'L_sh1_study_x', 'L_sh1_study_y', 'L_sh1_study_z', 
                        'L_calc_study_x', 'L_calc_study_y', 'L_calc_study_z', 'L_5meta_study_x', 'L_5meta_study_y', 
                        'L_5meta_study_z', 'L_toe_study_x', 'L_toe_study_y', 'L_toe_study_z', 'r_shoulder_study_x', 
                        'r_shoulder_study_y', 'r_shoulder_study_z', 'L_shoulder_study_x', 'L_shoulder_study_y', 
                        'L_shoulder_study_z', 'SJN_x', 'SJN_y', 'SJN_z', 'C7_study_x', 'C7_study_y', 'C7_study_z', 
                        'TV8_x', 'TV8_y', 'TV8_z', 'STRN_x', 'STRN_y', 'STRN_z', 'TV12_x', 'TV12_y', 'TV12_z', 
                        'r_lelbow_study_x', 'r_lelbow_study_y', 'r_lelbow_study_z', 'r_melbow_study_x', 'r_melbow_study_y', 
                        'r_melbow_study_z', 'RUArm_x', 'RUArm_y', 'RUArm_z', 'r_mwrist_study_x', 'r_mwrist_study_y', 
                        'r_mwrist_study_z', 'r_lwrist_study_x', 'r_lwrist_study_y', 'r_lwrist_study_z', 'RForearm_x', 
                        'RForearm_y', 'RForearm_z', 'RHL2_x', 'RHL2_y', 'RHL2_z', 'RHM5_x', 'RHM5_y', 'RHM5_z', 'RHand_x', 
                        'RHand_y', 'RHand_z', 'L_lelbow_study_x', 'L_lelbow_study_y', 'L_lelbow_study_z', 'L_melbow_study_x', 
                        'L_melbow_study_y', 'L_melbow_study_z', 'LUArm_x', 'LUArm_y', 'LUArm_z', 'L_mwrist_study_x', 
                        'L_mwrist_study_y', 'L_mwrist_study_z', 'L_lwrist_study_x', 'L_lwrist_study_y', 'L_lwrist_study_z', 
                        'LForearm_x', 'LForearm_y', 'LForearm_z', 'LHL2_x', 'LHL2_y', 'LHL2_z', 'LHM5_x', 'LHM5_y', 'LHM5_z', 
                        'LHand_x', 'LHand_y', 'LHand_z', 'RHD_x', 'RHD_y', 'RHD_z', 'LHD_x', 'LHD_y', 'LHD_z', 'FHD_x', 'FHD_y', 
                        'FHD_z', 'BHD_x', 'BHD_y', 'BHD_z']

    df = df[order_mks_must_have]

    df.to_csv(os.path.join(os.path.dirname(data_path), "new_file.csv"))

elif os.path.basename(data_path) == "3d_keypoints_filtered.csv":

    df = pd.read_csv(data_path)

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
    
    df = df[order_jcp_must_have]

    df.to_csv(os.path.join(os.path.dirname(data_path), "new_file_2.csv"))