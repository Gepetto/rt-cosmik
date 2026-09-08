 

import os
import pandas as pd
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat_shapes

import yaml, math

from utils.reader_parameters import parse_params, convert_to_class
 
from utils.data_elaboration_utils import  read_keypoints_from_csv,create_virtual_head_markers, init_kf_state, kf_keypoints_step, to_utc, fuse_stereo_keypoints, kabsch_global, estimate_joint_axis_accel_variances,build_per_joint_Q, remap_hpe_to_mocap, plot_ref_est_ft_concatenated, mpjpe, score_to_rgba, plot_3D_keypoints_and_scores
from utils.model_utils_motif import Robot, build_biomechanical_model 

from utils.vizutils import display_model_frames, box_between_frames, set_tf, draw_table, display_com, compute_cop, display_force, addViewerBox, addViewerSphere, applyViewerConfiguration

import matplotlib.pyplot as plt
 
from example_robot_data import load 


from scipy.signal import butter, sosfiltfilt


def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))


subject_maj= "Zoe"
subject_min = "zoe"
id_test = '4665'
task = "sanding"

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))

# load parameters class defined in parameters.toml file
dict_param = parse_params(os.path.join(script_directory, 'parameters.toml'))
param = convert_to_class(dict_param) # define class param 


###### Setup model (state the dof of interest from the general 37dof human model described in the documentation using active_joints)
 
#Load general human model 
base_path = "/root/workspace/ros_ws/src/rt-cosmik"
urdf_name =  "human.urdf"#f"generated_{gender}_mass{weigth}_size{size}.urdf"

urdf_path = os.path.join('/root/workspace/ros_ws/src/rt-cosmik/cosmik_vizu/model/human_urdf/urdf', urdf_name)
print(urdf_path)
# urdf_meshes_path = os.path.join(base_path, "meshes")
urdf_meshes_path = "/root/workspace/ros_ws/src/rt-cosmik/cosmik_vizu/model/"
print(urdf_meshes_path)
robot=Robot(urdf_path,urdf_meshes_path, param.free_flyer) 
model, collision_model,  visual_model, param=build_biomechanical_model(robot, param) 
data = model.createData()


model_mocap, collision_model_mocap,  visual_model_mocap, param=build_biomechanical_model(robot, param) 
data_mocap = model.createData()

# --- make visual_model_mocap grey ---
for go in visual_model_mocap.geometryObjects:
    go.overrideMaterial = True                     # ignore URDF materials/textures
    go.meshColor = np.array([0.4, 0.4, 0.4, 0.7])  # RGBA in [0,1]; e.g., mid-grey

# load panda model 

robot = load("panda")
model_robot = robot.model

visual_model_robot = robot.visual_model
collision_model_robot = robot.collision_model
data_robot = robot.data


### Load human data
csv_path = f"{base_path}/output/mocap/mocap_{subject_maj}/{task}/q_mocap_downsampled.csv"  # your file
q_ref = pd.read_csv(csv_path).to_numpy(dtype=float)  # shape: (num_samples, num_joints_including_FF)

csv_path = f"{base_path}/output/cosmik/cosmik_{subject_min}/{task}/q_cosmik_swika.csv"  # your file
#csv_path = "data_for_vid_cosmik/q_cosmik_swika.csv"  # your file
q_est = pd.read_csv(csv_path).to_numpy(dtype=float)  # shape: (num_samples, num_joints_including_FF)

csv_path = f"{base_path}/output/robot/robot_data_interpolated.csv"  # your file
df = pd.read_csv(csv_path, parse_dates=["_cam_time", "timestamp"])

# determine the synchro between the robot and the cam data
t_cam   = pd.read_csv(f"{base_path}/output/4687/mouv/{task}/camera_0_timestamps.csv",   parse_dates=["timestamp"])
t_robot = pd.read_csv(csv_path, parse_dates=["timestamp"])


t_cam["timestamp"]   = to_utc(t_cam["timestamp"])
t_robot["timestamp"] = to_utc(t_robot["timestamp"])

t_cam = t_cam.reset_index().rename(columns={"index":"cam_idx"})
t_robot = t_robot.reset_index().rename(columns={"index":"robot_idx"})

exact = t_cam.merge(t_robot, on="timestamp", how="inner")
if not exact.empty:
    first = exact.sort_values("timestamp").iloc[0]
    print("Exact match:")
    print("timestamp:", first["timestamp"])
    print("cam_idx:", int(first["cam_idx"]), "robot_idx:", int(first["robot_idx"]))
else:
    # 2) Nearest match within a tolerance (e.g., 5 ms)
    tol = pd.Timedelta("5ms")
    cam_s   = t_cam.sort_values("timestamp")
    t_robot_s = t_robot.sort_values("timestamp")
    nearest = pd.merge_asof(cam_s, t_robot_s, on="timestamp", direction="nearest", tolerance=tol, suffixes=("_cam","_robot"))
    nearest = nearest.dropna(subset=["robot_idx"])
    if not nearest.empty:
        first = nearest.iloc[0]
        print(f"Nearest match within {tol}:")
        print("timestamp:", first["timestamp"])
        print("cam_idx:", int(first["cam_idx"]), "robot_idx:", int(first["robot_idx"]))
    else:
        print("No match found (even within tolerance).")

pos_cols = [f"position.panda_joint{i}" for i in range(1, 8)]
q_robot = df[pos_cols].to_numpy(dtype=float)
q_robot = np.hstack([q_robot, np.zeros((q_robot.shape[0], 2), dtype=q_robot.dtype)]) # add zeros to fixed finger joints
# for jid in range(1, model_robot.njoints):
#     print(f"[{jid}] {model_robot.names[jid]}")

# load the base configuration of the panda robot 
with open("cosmik_vizu/robot_base_pose.yaml") as f:
    Y = yaml.safe_load(f)["world_T_robot"]
R = np.array(Y["rotation_matrix"], dtype=float)
t = np.array(Y["translation"], dtype=float)
T = pin.SE3(R, t)
T_robot=np.eye(4)
T_robot[0:3,0:3]=R
T_robot[0:3,3]=t

 



## load trianulated non filtred keypoints using left side camera
csv_path = f"{base_path}/output/cosmik_jcp/{subject_maj}/{task}/{task}_3d_keypoints.csv"   
keypoints_02, joints = read_keypoints_from_csv(csv_path)

csv_path = f"{base_path}/output/cosmik_jcp/{subject_maj}/{task}/{task}_3d_keypoints46.csv"    
keypoints_46, joints = read_keypoints_from_csv(csv_path)



 


## load confidence scor for each cam

df0 = pd.read_csv(f"{base_path}/output/output_2d/{subject_maj}/{task}/{task}_scores_0.csv")
df2 = pd.read_csv(f"{base_path}/output/output_2d/{subject_maj}/{task}/{task}_scores_2.csv")

# rename score cols to keep them distinct, then merge on 'frame'
scores0 = [c for c in df0.columns if c.endswith("_score")]
scores2 = [c for c in df2.columns if c.endswith("_score")]
scores0=scores0[1:]# remove the first index cause it is mean_score
scores2=scores2[1:]
df0_ren = df0.rename(columns={c: f"{c}_c0" for c in scores0})
df1_ren = df2.rename(columns={c: f"{c}_c2" for c in scores2})

df_merged = pd.merge(df0_ren[["frame"] + [f"{c}_c0" for c in scores0]],
                     df1_ren[["frame"] + [f"{c}_c2" for c in scores2]],
                     on="frame", how="inner")

# example: arrays per camera (aligned by frame)
scores_cam0 = df_merged[[f"{c}_c0" for c in scores0]].to_numpy(float)
scores_cam2 = df_merged[[f"{c}_c2" for c in scores2]].to_numpy(float)

df4 = pd.read_csv(f"{base_path}/output/output_2d/{subject_maj}/{task}/{task}_scores_4.csv")
df6 = pd.read_csv(f"{base_path}/output/output_2d/{subject_maj}/{task}/{task}_scores_6.csv")

# rename score cols to keep them distinct, then merge on 'frame'
scores4 = [c for c in df4.columns if c.endswith("_score")]
scores6 = [c for c in df6.columns if c.endswith("_score")]
scores4=scores4[1:]# remove the first index cause it is mean_score
scores6=scores6[1:]
df4_ren = df4.rename(columns={c: f"{c}_c4" for c in scores4})
df6_ren = df6.rename(columns={c: f"{c}_c6" for c in scores6})

df_merged = pd.merge(df4_ren[["frame"] + [f"{c}_c4" for c in scores4]],
                     df6_ren[["frame"] + [f"{c}_c6" for c in scores6]],
                     on="frame", how="inner")

# example: arrays per camera (aligned by frame)
scores_cam4 = df_merged[[f"{c}_c4" for c in scores4]].to_numpy(float)
scores_cam6 = df_merged[[f"{c}_c6" for c in scores6]].to_numpy(float)

# load JCP from mocap 
path = f"{base_path}/output/mocap_jcp/{subject_maj}/{task}"
csv_path = f"{path}/joint_center_positions.csv"   
jcp, joints = read_keypoints_from_csv(csv_path,1/1000)

csv_path =f"{base_path}/output/mocap/mocap_{subject_maj}/{task}/mocap_downsampled_to_40hz.csv"
mks, joints = read_keypoints_from_csv(csv_path,1/1000)

selected = ["FHD", "LHD", "RHD"]
indices = [joints.index(j) for j in selected if j in joints]

# Slice the keypoints
selected_mks = mks[:, indices, :]   # shape (N, 3, 3)

jcp_extended = np.concatenate([jcp, selected_mks], axis=1)

# position of the cameras in the world frame (please add a function for that)
transformation_file = f"{base_path}//config/cam_params/{subject_maj}/calib_mocap_2_cam0/soder.txt"
R_0, d_0, _,_ = load_transformation(transformation_file)
T_c0 = np.eye(4)
T_c0[:3, :3] = R_0
T_c0[:3, 3]  = d_0.reshape(3)

transformation_file = f"{base_path}//config/cam_params/{subject_maj}/calib_mocap_2_cam2/soder.txt"
R_2, d_2, _,_ = load_transformation(transformation_file)
T_c2= np.eye(4)
T_c2[:3, :3] = R_2
T_c2[:3, 3]  = d_2.reshape(3)

transformation_file = f"{base_path}//config/cam_params/{subject_maj}/calib_mocap_2_cam4/soder.txt"
R_4, d_4, _,_ = load_transformation(transformation_file)
T_c4 = np.eye(4)
T_c4[:3, :3] = R_4
T_c4[:3, 3]  = d_4.reshape(3)

transformation_file = f"{base_path}//config/cam_params/{subject_maj}/calib_mocap_2_cam6/soder.txt"
R_6, d_6, _,_ = load_transformation(transformation_file)
T_c6 = np.eye(4)
T_c6[:3, :3] = R_6
T_c6[:3, 3]  = d_6.reshape(3)

# Filter keypoints using kalman filter 

# Indices for your 26-joint layout (match your CSV order)
IDX_hpe = {
    "Nose":0,"LEye":1,"REye":2,"LEar":3,"REar":4,
    "LShoulder":5,"RShoulder":6,"LElbow":7,"RElbow":8,"LWrist":9,"RWrist":10,
    "LHip":11,"RHip":12,"LKnee":13,"RKnee":14,"LAnkle":15,"RAnkle":16,
    "Head":17,"Neck":18,"midHip":19,
    "LBigToe":20,"RBigToe":21,"LSmallToe":22,"RSmallToe":23,"LHeel":24,"RHeel":25,
}

# A light set of "bones" (pairs of indices) to keep lengths roughly constant
BONES = [
    (IDX_hpe["Head"], IDX_hpe["Neck"]),
    (IDX_hpe["Neck"], IDX_hpe["midHip"]),
    (IDX_hpe["Neck"], IDX_hpe["LShoulder"]), (IDX_hpe["LShoulder"], IDX_hpe["LElbow"]), (IDX_hpe["LElbow"], IDX_hpe["LWrist"]),
    (IDX_hpe["Neck"], IDX_hpe["RShoulder"]), (IDX_hpe["RShoulder"], IDX_hpe["RElbow"]), (IDX_hpe["RElbow"], IDX_hpe["RWrist"]),
    (IDX_hpe["midHip"], IDX_hpe["LHip"]), (IDX_hpe["LHip"], IDX_hpe["LKnee"]), (IDX_hpe["LKnee"], IDX_hpe["LAnkle"]),
    (IDX_hpe["midHip"], IDX_hpe["RHip"]), (IDX_hpe["RHip"], IDX_hpe["RKnee"]), (IDX_hpe["RKnee"], IDX_hpe["RAnkle"]),
    (IDX_hpe["LAnkle"], IDX_hpe["LHeel"]), (IDX_hpe["RAnkle"], IDX_hpe["RHeel"]),
    (IDX_hpe["LAnkle"], IDX_hpe["LBigToe"]), (IDX_hpe["RAnkle"], IDX_hpe["RBigToe"]),
]

# dictionnary to map hpe index to the mocap ones for comparison/plotting and parameters tuning
HPE2_MOCAP={# key is hpe idenx value if mcoap index
    "RShoulder":[6,0],
    "LShoulder":[5,1],
    "LElbow":[7, 4],
    "RElbow":[8, 3],
    "LWrist":[9,6 ],
    "RWrist":[10, 5],
    "LHip":[11, 8],
    "RHip":[12, 7],
    "LKnee":[13, 11],
    "RKnee":[14, 10],
    "LAnkle":[15, 13],
    "RAnkle":[16, 12],
    "Head":[17,  ],
    "Neck":[18, 2],
    "midHip":[19, 9],
    "LBigToe":[20, 17],
    "RBigToe":[21, 16],
    "LSmallToe":[22, 19],
    "RSmallToe":[23, 18],
    "LHeel":[24, 15],
    "RHeel":[25, 14],
    "LEar":[3,  ],
    "REar":[4,  ]
}



## plot 3D keypoints and their score
#plot_3D_keypoints_and_scores(IDX_hpe,scores_cam0,scores_cam2)


step=1 # step between samples
i0=0 # initial elaboration/display sample



# 

# initial 3D keypoint estimate using only one side  NO FUSION !
keypoints=[]
for t in range(i0,len(q_ref),step):#
    keypoints_frame, missing = remap_hpe_to_mocap(keypoints_02[t,:,:], HPE2_MOCAP, J_out=20)
 
    keypoints.append(keypoints_frame)

keypoints=np.array(keypoints) 

R, t=kabsch_global(keypoints, jcp, weights=None)
keypoints = (keypoints @ R.T) + t # keypoints are aligned with mocap




#Fuse data

jcp_hpe=[]
scores_hpe=[]
keypoints_fused=[]
gamma=5.7
for t in range(i0,len(q_ref),step):#

        
    keypoints_fused_frame, scores_fused = fuse_stereo_keypoints(
        np.asarray(keypoints_02[t,:,:], dtype=float), np.asarray(keypoints_46[t,:,:], dtype=float),                   # shape (26,3) each
        scores_cam0[t,:], scores_cam2[t,:],   # shape (26,)
        scores_cam4[t,:], scores_cam6[t,:],   # shape (26,)
        T_c0, T_c2, T_c4, T_c6,     # 4x4 each
        sigma=None,                 # or set e.g. sigma=1.0
        gamma=gamma
)
    # calculate JCP and SCORES in mocap order and remove head
    jcp_hpe_frame, missing = remap_hpe_to_mocap(keypoints_fused_frame, HPE2_MOCAP, J_out=20)

    pairs = [(v[0], v[1]) for v in HPE2_MOCAP.values() if len(v) == 2]          # (hpe_idx, mocap_idx)
    hpe_idx_sorted_by_mocap = [h for h, m in sorted(pairs, key=lambda t: t[1])]  # order by mocap_idx    
    scores_hpe_frame = scores_fused[np.array(hpe_idx_sorted_by_mocap)] 


    jcp_hpe.append(jcp_hpe_frame)
    scores_hpe.append(scores_hpe_frame)
    keypoints_fused.append(keypoints_fused_frame)

jcp_hpe=np.array(jcp_hpe) 
scores_hpe=np.array(scores_hpe) 
keypoints_fused=np.array(keypoints_fused) 


#solve procruste problem to align keypoints and jcp 
R, t=kabsch_global(jcp_hpe, jcp, weights=None)
jcp_hpe = (jcp_hpe @ R.T) + t

keypoints_fused= (keypoints_fused @ R.T) + t #us eteh same transformation for 26 keypoints


## Kalman filtering of the jcp_hpe

# distances for each bone in the first frame
idx_LARM=[HPE2_MOCAP["LShoulder"][1],HPE2_MOCAP["LElbow"][1],HPE2_MOCAP["LWrist"][1]]
idx_RARM=[HPE2_MOCAP["RShoulder"][1],HPE2_MOCAP["RElbow"][1],HPE2_MOCAP["RWrist"][1]]

ref_len = np.zeros(4, dtype=float) # only the arms

ref_len[0] = np.linalg.norm(jcp_hpe[0,idx_LARM[0],:]-jcp_hpe[0,idx_LARM[1],:]     ) #left upper arm
ref_len[1] = np.linalg.norm(jcp_hpe[0,idx_LARM[1],:]-jcp_hpe[0,idx_LARM[2],:]     ) #left lower arm

ref_len[2] = np.linalg.norm(jcp_hpe[0,idx_RARM[0],:]-jcp_hpe[0,idx_RARM[1],:]     ) #rigth upper arm
ref_len[3] = np.linalg.norm(jcp_hpe[0,idx_RARM[1],:]-jcp_hpe[0,idx_RARM[2],:]     ) #rigth lower arm

# print(ref_len)


# ref_len[0] = np.linalg.norm(jcp[0,idx_LARM[0],:]-jcp[0,idx_LARM[1],:]     ) #left upper arm
# ref_len[1] = np.linalg.norm(jcp[0,idx_LARM[1],:]-jcp[0,idx_LARM[2],:]     ) #left lower arm

# ref_len[2] = np.linalg.norm(jcp[0,idx_RARM[0],:]-jcp[0,idx_RARM[1],:]     ) #rigth upper arm
# ref_len[3] = np.linalg.norm(jcp[0,idx_RARM[1],:]-jcp[0,idx_RARM[2],:]     ) #rigth lower arm

# print(ref_len)



# initialization of Kalman filter 
r2=(2e-2/1.5960)*(2e-2/1.5960) # assume 2cm error

state = init_kf_state(20, dt=step/40,q=1e-3, r=r2,
                ref_len=ref_len,       # length for arms
                bone_alpha=0.1, n_proj_iters=3)

#optimal tuning of R measurement noise cov matrix
# usefull to set marker specific covariance
state["Rj_init"] = np.tile(state["R"][None, :, :], (state["K"], 1, 1))

err = jcp - jcp_hpe                                # (N, 20, 3)
e = err - err.mean(axis=0, keepdims=True)          # demean over time
den = max(jcp.shape[0] - 1, 1)                     # avoid div by zero
Rj = np.einsum('nkd,nke->kde', e, e) / den         # (20, 3, 3)

# small regularization for numerical stability (optional)
Rj += 1e-6 * np.eye(3)[None, :, :]



state["Rj"] = ( Rj + state["Rj_init"])/(3) #+ 0.01/np.mean(scores_hpe)

 
 


#optimal tuning of Q process noise cov matrix
# mocap positions as reference (meters)
var_ax = estimate_joint_axis_accel_variances(jcp, dt=step/40, shrink=0.1)
Qj = build_per_joint_Q(var_ax, dt=step/40)

state["Qj"] = Qj#  # attach to KF state

 
jcp_hpe_filtered=[]
for t in range(i0,len(q_ref),step):#
     
    # state["Rj"] = state.get("Rj_init2", np.tile(R[None, :, :], (K, 1, 1)))

    # for j in range(K):
    #     state["Rj"][j] = state["Rj"][j]* (0.0 / (scores_hpe[j] + .9))
        
    # For each new sample z_t (K,3):
    jcp_hpe_filtered_frame = kf_keypoints_step(jcp_hpe[t,:,:], state, HPE2_MOCAP)   # (K,3)
    jcp_hpe_filtered.append(jcp_hpe_filtered_frame)
    
jcp_hpe_filtered=np.array(jcp_hpe_filtered)    

# C7_t=jcp_hpe_filtered[:,HPE2_MOCAP["Neck"][1],:]
# RSHO_t=jcp_hpe_filtered[:,HPE2_MOCAP["RShoulder"][1],:]
# LSHO_t=jcp_hpe_filtered[:,HPE2_MOCAP["LShoulder"][1],:]

# vR_t, vL_t = make_virtual_shoulders(C7_t, RSHO_t, LSHO_t, up_offset=0.01, ap_offset=0.01)
 
# jcp_hpe_filtered[:,HPE2_MOCAP["RShoulder"][1],:]=vR_t
# jcp_hpe_filtered[:,HPE2_MOCAP["LShoulder"][1],:]=vL_t

 
head=keypoints_fused[:,HPE2_MOCAP["Head"][0],:]
lear=keypoints_fused[:,HPE2_MOCAP["LEar"][0],:]
rear=keypoints_fused[:,HPE2_MOCAP["REar"][0],:]

FHD, LHD, RHD=create_virtual_head_markers(head, lear, rear,
                                ear_up=0.06,  ear_ap=-0.00,
                                head_down=-0.02, head_ap=0.00)

jcp_hpe_filtered = np.concatenate(
    [jcp_hpe_filtered, FHD[:, None, :], LHD[:, None, :], RHD[:, None, :]], axis=1
)




 
dt = step/40          # seconds per sample (set yours)
fs = 1.0 / dt
fc = 4.0           # cutoff frequency [Hz] (set yours)
order = 5

# Stable SOS form + zero-phase filtering
sos = butter(order, fc / (fs * 0.5), btype='low', output='sos')
jcp_hpe_filtered = sosfiltfilt(sos, jcp_hpe_filtered, axis=0)  # zero phase, no la

point_names = [
    "RShoulder", "LShoulder", "Neck",  "RElbow", "LElbow", 
    "RWrist", "LWrist", "RHip", "LHip", "midHip",
    "RKnee", "LKnee", "RAnkle", "LAnkle","RHeel", "LHeel",
    "RBigToe", "LBigToe", "RSmallToe", "LSmallToe", "FHD","LHD","RHD"
]

# Flatten each frame into a single row (frames, points*3)
frames, n_points, coords = jcp_hpe_filtered.shape
df = pd.DataFrame(jcp_hpe_filtered.reshape(frames, -1))

# Generate column names: point_x, point_y, point_z
columns = []
for name in point_names:
    columns += [f'{name}_x', f'{name}_y', f'{name}_z']
df.columns = columns

# Save CSV
df.to_csv(f'{base_path}/output/cosmik_jcp/{subject_maj}/{task}/{task}_jcp_hpe_filtered_2.csv', index=False)
print("data_saved")

#R, t=kabsch_global(jcp_hpe_filtered, jcp, weights=None)
#jcp_hpe_filtered = (jcp_hpe_filtered @ R.T) + t

print(jcp_hpe_filtered.shape)
# err=mpjpe(jcp, jcp_hpe_filtered)      
# print(err)

# print("err_shoulder fused") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RShoulder"][1],:] - jcp_hpe[:,HPE2_MOCAP["RShoulder"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LShoulder"][1],:] - jcp_hpe[:,HPE2_MOCAP["LShoulder"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)


# print("err_shoulder fused-filter") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RShoulder"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["RShoulder"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LShoulder"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["LShoulder"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# print("err_elbow fused") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RElbow"][1],:] - jcp_hpe[:,HPE2_MOCAP["RElbow"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LElbow"][1],:] - jcp_hpe[:,HPE2_MOCAP["LElbow"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# print("err_elbow fused-filter") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RElbow"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["RElbow"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LElbow"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["LElbow"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)



# print("err_wrist fused") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RWrist"][1],:] - jcp_hpe[:,HPE2_MOCAP["RWrist"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LWrist"][1],:] - jcp_hpe[:,HPE2_MOCAP["LWrist"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# print("err_wrist_filter fused-filter") 
# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["RWrist"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["RWrist"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)

# rms =  np.sqrt( np.sum((jcp[:,HPE2_MOCAP["LWrist"][1],:] - jcp_hpe_filtered[:,HPE2_MOCAP["LWrist"][1],:])**2) /len(jcp) )  # (N, J)
# print(rms)




# est_len[:,0] = np.linalg.norm(jcp_hpe_filtered[:,idx_LARM[0],:]-jcp_hpe_filtered[:,idx_LARM[1],:], axis=1    ) #left upper arm
# est_len[:,1] = np.linalg.norm(jcp_hpe_filtered[:,idx_LARM[1],:]-jcp_hpe_filtered[:,idx_LARM[2],:], axis=1     ) #left lower arm

# est_len[:,2] = np.linalg.norm(jcp_hpe_filtered[:,idx_RARM[0],:]-jcp_hpe_filtered[:,idx_RARM[1],:], axis=1     ) #rigth upper arm
# est_len[:,3] = np.linalg.norm(jcp_hpe_filtered[:,idx_RARM[1],:]-jcp_hpe_filtered[:,idx_RARM[2],:], axis=1     ) #rigth lower arm


# # est_len[:,0] = np.linalg.norm(jcp[:,idx_LARM[0],:]-jcp[:,idx_LARM[1],:], axis=1    ) #left upper arm
# # est_len[:,1] = np.linalg.norm(jcp[:,idx_LARM[1],:]-jcp[:,idx_LARM[2],:], axis=1     ) #left lower arm

# # est_len[:,2] = np.linalg.norm(jcp[:,idx_RARM[0],:]-jcp[:,idx_RARM[1],:], axis=1     ) #rigth upper arm
# # est_len[:,3] = np.linalg.norm(jcp[:,idx_RARM[1],:]-jcp[:,idx_RARM[2],:], axis=1     ) #rigth lower arm



# # t = np.arange(N)

# # fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

# # # Left arm subplot
# # ax = axes[0]
# # ax.plot(t, est_len[:, 0], label="Upper")
# # ax.plot(t, est_len[:, 1], label="Lower")
# # ax.axhline(ref_len[0], linestyle="--", linewidth=1, label="Ref upper")
# # ax.axhline(ref_len[1], linestyle="--", linewidth=1, label="Ref lower")
# # ax.set_title("Left arm")
# # ax.set_xlabel("Frame")
# # ax.set_ylabel("Length (m)")
# # ax.legend(frameon=False)

# # # Right arm subplot
# # ax = axes[1]
# # ax.plot(t, est_len[:, 2], label="Upper")
# # ax.plot(t, est_len[:, 3], label="Lower")
# # ax.axhline(ref_len[2], linestyle="--", linewidth=1, label="Ref upper")
# # ax.axhline(ref_len[3], linestyle="--", linewidth=1, label="Ref lower")
# # ax.set_title("Right arm")
# # ax.set_xlabel("Frame")
# # ax.legend(frameon=False)

# # plt.tight_layout()
# # plt.show()


# #err=mpjpe(jcp[:,HPE2_MOCAP["LElbow"][1],:], jcp_hpe_filtered[:,HPE2_MOCAP["LElbow"][1],:])      
 
 
# # keypoints_fused_all=[]

# # #for i in range(i0,i0+100,step):#len(q_ref),step):#
# # for t in range(i0,len(q_ref),step):#
# #     keypoints_fused = fuse_stereo_keypoints(
# #         np.asarray(keypoints_02[t,:,:], dtype=float), np.asarray(keypoints_46[t,:,:], dtype=float),                   # shape (26,3) each
# #         scores_cam0[t,:], scores_cam2[t,:],   # shape (26,)
# #         scores_cam4[t,:], scores_cam6[t,:],   # shape (26,)
# #         T_c0, T_c2, T_c4, T_c6,     # 4x4 each
# #         sigma=None,                 # or set e.g. sigma=1.0
# #         gamma=5.799999999999999
# # )
    
# #     # For each new sample z_t (K,3):
# #     keypoints_fused = kf_keypoints_step(keypoints_fused[:,:], state)   # (K,3)
# #     keypoints_fused_all.append(keypoints_fused)
    
# # keypoints_fused_all=np.array(keypoints_fused_all)   
 
# # jcp_hpe = np.zeros((keypoints_fused_all.shape[0], jcp.shape[1], 3))

# # for joint_name, indices in HPE2_MOCAP.items():
# #     if len(indices) < 2:
# #         continue  # skip if mocap index missing (like "Head")
# #     hpe_idx, jcp_idx = indices
# #     jcp_hpe[:, jcp_idx, :] = keypoints_fused_all[:, hpe_idx, :]
    
# # ## solve procruste problem to align keypoints and jcp 
# # R, t=kabsch_global(jcp_hpe, jcp, weights=None)    
# # jcp_hpe_ft = (jcp_hpe @ R.T) + t

    
  
 
plot_ref_est_ft_concatenated(
    jcp,          # (N, J_ref, 3)  e.g., jcp (mocap)
    keypoints,          # (N, J_est, 3)  e.g., keypoints_fused_all (HPE order)
    jcp_hpe_filtered,           # (N, J_ft,  3)  fine-tuned version (see ft_in_ref_order)
    HPE2_MOCAP,      # dict: name -> [hpe_idx, mocap_idx]
    ft_in_ref_order=None,   # True if `ft` is already arranged in ref/mocap order
    title=None)




# N = jcp.shape[0]
# hpe_arr = np.full((N, 26, 3), np.nan, dtype=float)  # target in HPE order

# # Fill HPE joints that have a mocap index
# for name, idxs in HPE2_MOCAP.items():
#     if len(idxs) >= 2 and idxs[1] is not None:
#         hpe_idx, mocap_idx = idxs[0], idxs[1]
#         hpe_arr[:, hpe_idx, :] = jcp_hpe_filtered[:, mocap_idx, :]

# hpe_arr[:,0,:]=keypoints_fused[:,0,:] #nose
# hpe_arr[:,1,:]=keypoints_fused[:,1,:] #Leye
# hpe_arr[:,2,:]=keypoints_fused[:,2,:] #Reye
# hpe_arr[:,3,:]=keypoints_fused[:,3,:] #Lear
# hpe_arr[:,4,:]=keypoints_fused[:,4,:] #Rear
# hpe_arr[:,17,:]=keypoints_fused[:,17,:] #head
# print(hpe_arr)
 
# # keypoints_fused: (N, 26, 3) in HPE order, last axis = [x,y,z]
# cols = "Nose_x,Nose_y,Nose_z,LEye_x,LEye_y,LEye_z,REye_x,REye_y,REye_z,LEar_x,LEar_y,LEar_z,REar_x,REar_y,REar_z,LShoulder_x,LShoulder_y,LShoulder_z,RShoulder_x,RShoulder_y,RShoulder_z,LElbow_x,LElbow_y,LElbow_z,RElbow_x,RElbow_y,RElbow_z,LWrist_x,LWrist_y,LWrist_z,RWrist_x,RWrist_y,RWrist_z,LHip_x,LHip_y,LHip_z,RHip_x,RHip_y,RHip_z,LKnee_x,LKnee_y,LKnee_z,RKnee_x,RKnee_y,RKnee_z,LAnkle_x,LAnkle_y,LAnkle_z,RAnkle_x,RAnkle_y,RAnkle_z,Head_x,Head_y,Head_z,Neck_x,Neck_y,Neck_z,midHip_x,midHip_y,midHip_z,LBigToe_x,LBigToe_y,LBigToe_z,RBigToe_x,RBigToe_y,RBigToe_z,LSmallToe_x,LSmallToe_y,LSmallToe_z,RSmallToe_x,RSmallToe_y,RSmallToe_z,LHeel_x,LHeel_y,LHeel_z,RHeel_x,RHeel_y,RHeel_z".split(",")

# N = hpe_arr.shape[0]
# flat = hpe_arr.reshape(N, 26*3)  # [kp1_x,kp1_y,kp1_z, kp2_x,...]

# df = pd.DataFrame(flat, columns=cols)
# df.to_csv("3D_keypoints_fused.csv", index=False)
 
 
 
# animate models
 
step=1
i0=20#350 # initial elaboration/display sample

if param.viewer=="meshcat":

    print("meshcat animation...")
     
    # frame2display=["middle_lumbar_Z","right_hand"]
    # display_model_frames(model, visual_model, frame2display, param) 
    
    quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs()#set the human model uprigth
    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
       
      
    # one shared MeshCat viewer
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

    # estimated model
    # viz_est = MeshcatVisualizer(model, collision_model, visual_model)
    # viz_est.initViewer(viewer)
    # viz_est.loadViewerModel(rootNodeName="est")     # older versions

    # mocap/reference model
    viz_ref = MeshcatVisualizer(model_mocap, collision_model_mocap, visual_model_mocap)
    viz_ref.initViewer(viewer)
    viz_ref.loadViewerModel(rootNodeName="ref")

    # panda model
    viz_robot = MeshcatVisualizer(model_robot, collision_model_robot, visual_model_robot)
    viz_robot.initViewer(viewer)
    viz_robot.loadViewerModel(rootNodeName="panda")
    viz_robot.viewer["panda"].set_transform(T.homogeneous)

    # change background color
    native_viz = viz_ref.viewer
    native_viz["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
    native_viz["/Background"].set_property("bottom_color", [1, 1, 1])  # Same color → flat background
    grid_height = -0.0  # Negative z = lower the grid
    native_viz["/Grid"].set_transform(
    np.array([
        [1, 0, 0, 0],  # Rotation (identity)
        [0, 1, 0, 0],
        [0, 0, 1, grid_height],
        [0, 0, 0, 1]
    ]))
     
   
    
    # add visual for forces plates   
    fp_names = ["forceplate1", "forceplate2","forceplate3", "forceplate4"]   
    # meters, (x, y) per force plate
    fp_dim = [
        (0.5, 0.6),  # FP1
        (0.50, 0.60),  # FP2
        (0.50, 0.60),  # FP3
        (0.9, 1.8),  # FP4
        (0.5, 0.6),  # FP5
        
    ]
    fp_centers = [
    ( -0.830,  -0.3, 0.0),# FP1
    ( -0.25,  -0.3, 0.0),# FP2
    ( 0.39,  -0.3, 0.0),# FP3
    ( -1.68,  -0.3, 0.0),# FP4
    ( -0.25,  0.3, 0.0),# FP5
    ]

    for j, ((sx, sy), (cx, cy, cz)) in enumerate(zip(fp_dim, fp_centers), start=1):
        name = f"force_plate_{j}"
        addViewerBox(viz_robot, name, sx, sy, 0.01, rgba=[0.5, 0.5, 0.5, 1.0]) # create a box

        # build world transform (centered box: put center at (cx,cy,cz))
        T = np.eye(4)
        T[:3, 3] = [cx, cy, cz + 0.01/2.0]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
        set_tf(viz_robot, name, T)
    
    # # add visual for wall and windows
    
    # addViewerBox(viz_robot, "wall", 10, 0.5, 3, rgba=[0.96, 0.96, 0.86, 1.0]) # create a box
    # T = np.eye(4)
    # T[:3, 3] = [0,3, 1.5]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    # set_tf(viz_robot,  "wall", T)
  
  
    # addViewerBox(viz_robot, "window1", 2, 0.1, 1.5, rgba=[0.25, 0.25, 0.25, 1.0]) # create a box
    # T = np.eye(4)
    # T[:3, 3] = [-3,2.75, 1.8]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    # set_tf(viz_robot,  "window1", T)
  
    # addViewerBox(viz_robot, "window2", 2, 0.1, 1.5, rgba=[0.25, 0.25, 0.25, 1.0]) # create a box
    # T = np.eye(4)
    # T[:3, 3] = [3,2.75, 1.8]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    # set_tf(viz_robot,  "window2", T)
    
    # addViewerBox(viz_robot, "door", 1.2, 0.1, 2.2, rgba=[0.2, 0.2, 0.2, 1.0]) # create a box
    # T = np.eye(4)
    # T[:3, 3] = [0,2.75, 1.1]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    # set_tf(viz_robot,  "door", T)
    
    # Add frames
    T_name = np.eye(4)
    
    meshcat_shapes.textarea(viz_robot.viewer["R_world_text"], "R0", font_size=32)
    T_name[0,3]+=0.15
    T_name[1,3]+=0.15
    T_name[2,3]+=0.15
    set_tf(viz_robot,  "R_world_text", T_name)
    
    T_name = np.eye(4)
    meshcat_shapes.textarea(viz_robot.viewer["Rrobot_text"], "Rrobot", font_size=28)
    meshcat_shapes.textarea(viz_robot.viewer["R_c0_text"], "cam0", font_size=28)
    meshcat_shapes.textarea(viz_robot.viewer["R_c2_text"], "cam2", font_size=28)
    meshcat_shapes.textarea(viz_robot.viewer["R_c4_text"], "cam4", font_size=28)
    meshcat_shapes.textarea(viz_robot.viewer["R_c6_text"], "cam6", font_size=28)
    
    T_name = np.eye(4)
    #world frame
    meshcat_shapes.frame(
    viz_robot.viewer["R_world"],
    axis_length=0.4,
    axis_thickness=0.04,
    opacity=1,
    
    origin_radius=0.02) 
    T_name[1, 3]+=0.15  
   
    # robot frame
    meshcat_shapes.frame(
    viz_robot.viewer["R_robot"],
    axis_length=0.4,
    axis_thickness=0.02,
    opacity=1,
    origin_radius=0.02) 
    T_name[1, 3]+=0.15  
    set_tf(viz_robot,  "R_robot", T_robot)
    T_name[0:3,3]=T_robot[0:3,3]
    T_name[0,3]+=0.25
    T_name[1,3]-=0.25
    T_name[2,3]+=0.025
    set_tf(viz_robot,  "Rrobot_text", T_name)

    
    T_frame = np.eye(4)
    T_frame[0:3,0:3] =[[0,-1,0],
                       [-1,0,0],
                       [0,0,-1]]   
     
    for j  in range(5):
        fp_name = f'fp{j}'
        fp_frame = f'R_fp{j}'
        meshcat_shapes.textarea(viz_robot.viewer[fp_name], f'fp{j+1}', font_size=32)
        T_name[:3, 3] = fp_centers[j]
        T_name[2, 3]=0.03
        T_name[0, 3]=T_name[0, 3]+0.05   
        T_name[1, 3]=T_name[1, 3]+0.05     
        set_tf(viz_robot,  fp_name, T_name)

        
        meshcat_shapes.frame(
        viz_robot.viewer[fp_frame],
        axis_length=0.2,
        axis_thickness=0.02,
        opacity=0.8,
        origin_radius=0.02)
        T_frame[:3, 3] = fp_centers[j]
        T_frame[2, 3]=0.03   
        set_tf(viz_robot,  fp_frame, T_frame)

    # add cameras boxe and frames
    box_between_frames(viz_robot, "link_c0_c2", T_c0, T_c2,
                   thickness=0.1, height=0.1,
                   rgba=(0.01, 0.01, 0.01, 0.9))    
    
    
    
    T_cam=np.eye(4)
    T_cam[0:3,0:3] =[[1,0,0],
                       [0,0,-1],
                       [0,1,0]]  
    meshcat_shapes.frame(
    viz_robot.viewer["f_cam_0"],
    axis_length=0.2,
    axis_thickness=0.02,
    opacity=0.8,
    origin_radius=0.02) 
    set_tf(viz_robot,  "f_cam_0", T_c0)
    
    T_cam[0:3,3]=T_c0[0:3,3]
    T_cam[2,3]+=0.1
    set_tf(viz_robot,  "R_c0_text", T_cam)
    
    T_cam_0 = T_c0
    addViewerBox(viz_robot, "cam_0", 0.1, 0.1, 0.1 , rgba=[0.01, 0.01, 0.01, 1.0])
    set_tf(viz_robot,   "cam_0", T_cam_0)
    
    T_cam_2 = T_c2
    addViewerBox(viz_robot, "cam_2", 0.1, 0.1, 0.1 , rgba=[0.01, 0.01, 0.01, 1.0])
    set_tf(viz_robot,   "cam_2", T_cam_2)
    
    T_cam[0:3,3]=T_c2[0:3,3]
    T_cam[2,3]+=0.1
    set_tf(viz_robot,  "R_c2_text", T_cam)
    
    box_between_frames(viz_robot, "link_c4_c6", T_c4, T_c6,
                   thickness=0.1, height=0.1,
                   rgba=(0.01, 0.01, 0.01, 0.9))  
    
    meshcat_shapes.frame(
    viz_robot.viewer["f_cam_2"],
    axis_length=0.2,
    axis_thickness=0.02,
    opacity=0.8,
    origin_radius=0.02) 
    set_tf(viz_robot,  "f_cam_2", T_c2)
    
    
    meshcat_shapes.frame(
    viz_robot.viewer["f_cam_4"],
    axis_length=0.2,
    axis_thickness=0.02,
    opacity=0.8,
    origin_radius=0.02) 
    set_tf(viz_robot,  "f_cam_4", T_c4)
    T_cam[0:3,3]=T_c4[0:3,3]
    T_cam[2,3]+=0.1
    set_tf(viz_robot,  "R_c4_text", T_cam)
    
    
    meshcat_shapes.frame(
    viz_robot.viewer["f_cam_6"],
    axis_length=0.2,
    axis_thickness=0.02,
    opacity=0.8,
    origin_radius=0.02) 
    set_tf(viz_robot,  "f_cam_6", T_c6)
    T_cam[0:3,3]=T_c6[0:3,3]
    T_cam[2,3]+=0.1
    set_tf(viz_robot,  "R_c6_text", T_cam)
    
 
    addViewerBox(viz_robot, "cam_4", 0.1, 0.1, 0.1 , rgba=[0.01, 0.01, 0.01, 1.0])
    set_tf(viz_robot,   "cam_4", T_c4)
    
   
    addViewerBox(viz_robot, "cam_6", 0.1, 0.1, 0.1 , rgba=[0.01, 0.01, 0.01, 1.0])
    set_tf(viz_robot,   "cam_6", T_c6)
    
    # add table
    T_world_table = np.eye(4)
    T_world_table[:3, 3] = [0.9, -0.6, 0.0]

    draw_table(viz_robot, T_world_table) 
    
 
    images=[]
    for i in range(i0,i0+10,step):#len(q_ref),step):#
        
    #for i in range(i0,len(q_ref),step):#

        
        # for j, name in enumerate(names):
        #     col = IDX_hpe[name]
        #     #scores_frame = scores_cam2[i, col]+scores_cam3[i, col]
             
        #     sphere_name = f'keypoints_46_kf{j}'
        #     #rgba = score_to_rgba(scores_frame)       # e.g., scores_frame = s_fused[i]
          
        #     addViewerSphere(viz_est, sphere_name, 0.025, [1, 0, 0, 1])
        #     applyViewerConfiguration(viz_est, sphere_name, np.hstack((keypoints_46[i,j,:], np.array([0, 0, 0, 1]))))
     
    
        # for j  in range(keypoints_02.shape[1]):
        #     sphere_name = f'keypoints_02_{j}'
        #     addViewerSphere(viz_est, sphere_name, 0.025, [0, 1, 0, 1])
        #     applyViewerConfiguration(viz_est, sphere_name, np.hstack((keypoints_02[i,j,:], np.array([0, 0, 0, 1]))))
    
        # for j  in range(jcp_hpe.shape[1]):
        #     sphere_name = f'jcp_hpe{j}'
        #     addViewerSphere(viz_ref, sphere_name, 0.025, [1, 0, 0, 1])
        #     applyViewerConfiguration(viz_ref, sphere_name, np.hstack((jcp_hpe[i,j,:], np.array([0, 0, 0, 1]))))
      
        for j  in range(jcp_hpe_filtered.shape[1]):
            sphere_name = f'jcp_hpe_filtered{j}'
            addViewerSphere(viz_ref, sphere_name, 0.025, [0, 1, 0, 1])
            applyViewerConfiguration(viz_ref, sphere_name, np.hstack((jcp_hpe_filtered[i,j,:], np.array([0, 0, 0, 1]))))


        for j  in range(jcp.shape[1]):
            sphere_name = f'jcp_mocap{j}'
            
            addViewerSphere(viz_ref, sphere_name, 0.025, [0.5, 0.5, 0.5, 1])
            applyViewerConfiguration(viz_ref, sphere_name, np.hstack((jcp[i,j,:], np.array([0, 0, 0, 1]))))

        for j  in range(mks.shape[1]):
            sphere_name = f'mks{j}'
            
            addViewerSphere(viz_ref, sphere_name, 0.015, [1, 0, 0, 1])
            applyViewerConfiguration(viz_ref, sphere_name, np.hstack((mks[i,j,:], np.array([0, 0, 0, 1]))))

        
        
        mks
    
        #if i>=first["cam_idx"] and i<=len(q_robot): # because robot data starts after mocap and end before
        #    viz_robot.display(q_robot[i-first["cam_idx"],:])
        
        #viz_est.display(q_est[i,:])
        
        viz_ref.display(q_ref[i,:])
       
       # input()
        
        
 
        
        #images.append(viz_robot.viewer.get_image())

# imageio.mimsave("video.mp4", images, fps=40/step)
 
        
    
