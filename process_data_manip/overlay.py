import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.rtcosmik.camera.cam_utils import load_camera_parameters, load_cam_params
from src.rtcosmik.utils.read_write_utils import load_transformation, udp_csv_to_dataframe

#read transfo from nicolas file
def read_transformation_from_csv(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']

    # Extract rotation matrix R_mean
    R_start = lines.index('R_mean') + 1
    R = np.array([[float(x) for x in lines[R_start + i].split(',')] for i in range(3)])

    # Extract translation vector t_mean
    t_line = next(line for line in lines if line.startswith('t_mean'))
    t = np.array([float(x) for x in t_line.split(',')[1:]])  # skip 't_mean'

    return R, t

def project_and_draw_markers(frame, markers, rvec, tvec, K, D, color, scale=4, units_factor=1.0):
    for pt in markers:
        if np.isnan(pt).any():
            continue
        pt_scaled = (pt * units_factor).astype(np.float32).reshape(1, 1, 3)  # scale units if needed
        pt_2d, _ = cv2.projectPoints(pt_scaled, rvec, tvec, K, D)
        x, y = pt_2d[0][0]
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), scale, color, -1)

def draw_2d_keypoints_on_frame(frame, keypoints_2d, color=(255, 255, 0), scale=4):
    for x, y in keypoints_2d:
        if np.isnan([x, y]).any():
            continue
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), scale, color, -1)


# === Load  markers
no_trial = "Maxime"
task = "static"
path_to_csv_mocap = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"
marker_mocap_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']
df_mocap = udp_csv_to_dataframe(path_to_csv_mocap, marker_mocap_names)
assert df_mocap.shape[1] == len(marker_mocap_names) * 3
n_frames = len(df_mocap)
# Reshape into [n_frames, n_markers, 3]
marker_data_mocap = df_mocap.values.reshape(n_frames, len(marker_mocap_names), 3)


path_to_csv_lstm = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/augmented_markers_2.csv"
mks_names = [
           'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']

df_lstm = pd.read_csv(path_to_csv_lstm, skiprows=1)
assert df_lstm.shape[1] == len(mks_names) * 3
n_frames = len(df_lstm)
# Reshape into [n_frames, n_markers, 3]
marker_data_lstm = df_lstm.values.reshape(n_frames, len(mks_names), 3)


path_to_csv_kpts = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/3d_keypoints_filtered_2.csv"
mks_names_2 = [
    "Nose", "LEye", "REye", "LEar", "REar", 
    "LShoulder", "RShoulder", "LElbow", "RElbow", 
    "LWrist", "RWrist", "LHip", "RHip", 
    "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
    "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]
df_kpts = pd.read_csv(path_to_csv_kpts, skiprows=1)
assert df_kpts.shape[1] == len(mks_names_2) * 3
n_frames = len(df_kpts)
# Reshape into [n_frames, n_markers, 3]
marker_data_kpts = df_kpts.values.reshape(n_frames, len(mks_names_2), 3)



csv_path_2d= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/output_2d/{task}/{task}_camera_0.csv"
df_2d = pd.read_csv(csv_path_2d).iloc[:, 1:]
assert df_2d.shape[1] == len(mks_names_2)*2
kpt_2d = df_2d.values.reshape(len(df_2d), len(mks_names_2),2)



# === Load parameters
#with Procrustes_alignment
path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/Procrustes_alignment_results.csv"
# path= "/root/workspace/ros_ws/src/rt-cosmik/output/Maxime/results/Procrustes_alignment_results_static.csv"
# path="/root/workspace/ros_ws/src/rt-cosmik/output/Maxime/results_4cams/results/Procrustes_alignment_results_static.csv"
R, T = read_transformation_from_csv(path)
R = R.T #because its cam_to_mocap transfo
T = -R @ T

####if i want to use the transfo that we get with soder
base_path = "/root/workspace/ros_ws/src/rt-cosmik"
config_path = os.path.join(base_path, f"config/cam_params/{no_trial}")
R1, T1, s_trans, rms_error = load_transformation(os.path.join(config_path, "calib_mocap_2_cam0/soder.txt"))
R1 = R1.T #because its cam_to_mocap transfo
T1 = -R1 @ T1

R_total=R1 @ R
T_total = R1 @T  + T1

K, D = load_cam_params(os.path.join(config_path, "c0_params_color.yaml"))



# === Load video
video_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/camera_0.mp4"
cap = cv2.VideoCapture(video_path)
output_path = f"overlay_output_{task}.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

rvec, _ = cv2.Rodrigues(R)  
tvec = T.reshape(3, 1).astype(np.float32)


r = np.eye(3)
r,_ =cv2.Rodrigues(r)
t = np.zeros((3, 1))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(marker_data_lstm):
        break

    points3d = marker_data_lstm[frame_idx]  # (43, 3)
    project_and_draw_markers(frame, points3d, rvec, tvec, K, D, (0, 255, 0), scale=4, units_factor=1.0)

    points3d = marker_data_kpts[frame_idx]
    project_and_draw_markers(frame, points3d, rvec, tvec, K, D, (0, 0, 255), scale=4, units_factor=1.0)

    points3d = marker_data_mocap[frame_idx]
    project_and_draw_markers(frame, points3d, rvec, tvec, K, D, (255, 0, 0), scale=2, units_factor=1.0)

    draw_2d_keypoints_on_frame(frame, kpt_2d[frame_idx], color=(255, 255, 0), scale=3)

    writer.write(frame)
    cv2.imshow("Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
