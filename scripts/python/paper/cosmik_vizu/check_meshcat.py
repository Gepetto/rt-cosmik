 

import os
import pandas as pd
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat_shapes

import yaml, math

from utils.reader_parameters import parse_params, convert_to_class
 
from utils.data_elaboration_utils import  to_utc
from utils.model_utils_motif import Robot 

from utils.vizutils import box_between_frames, set_tf, draw_table, addViewerBox, addViewerSphere, applyViewerConfiguration

import matplotlib.pyplot as plt
 
from example_robot_data import load 

from src.rtcosmik.utils.read_write_utils import load_transformation

subject = "Zoe"
subject_mocap = "zoe"
id = '4687'
task = "robot_welding"
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))

###### Setup model (state the dof of interest from the general 37dof human model described in the documentation using active_joints)
#Load general human model 
base_path = "/root/workspace/ros_ws/src/rt-cosmik"
urdf_name =  "human_.urdf"#f"generated_{gender}_mass{weigth}_size{size}.urdf"

urdf_path = os.path.join('/root/workspace/ros_ws/src/rt-cosmik/cosmik_vizu/model/human_urdf/urdf', urdf_name)
urdf_meshes_path = "/root/workspace/ros_ws/src/rt-cosmik/cosmik_vizu/model/"

human=Robot(urdf_path,urdf_meshes_path) 
model_mocap = human.model
collision_model_mocap = human.collision_model
visual_model_mocap = human.visual_model
print(model_mocap.nq)
data_mocap = model_mocap.createData()

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
csv_path = f"{base_path}/output/mocap/mocap_{subject_mocap}/{task}/q_mocap_downsampled.csv"  # your file
q_ref = pd.read_csv(csv_path).to_numpy(dtype=float)  # shape: (num_samples, num_joints_including_FF)

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

# load the base configuration of the panda robot 
with open("cosmik_vizu/robot_base_pose.yaml") as f:
    Y = yaml.safe_load(f)["world_T_robot"]
R = np.array(Y["rotation_matrix"], dtype=float)
t = np.array(Y["translation"], dtype=float)
T = pin.SE3(R, t)
T_robot=np.eye(4)
T_robot[0:3,0:3]=R
T_robot[0:3,3]=t


# load JCP from mocap 
path = f"{base_path}/output/mocap_jcp/{subject_mocap}"
csv_path = f"{path}/{task}_joint_center_positions.csv"   
df = pd.read_csv(csv_path)
# get base joint names in order of first appearance
bases = []
for c in df.columns:
    if "_" in c:
        b, ax = c.rsplit("_", 1)
        if ax.lower() in ("x", "y", "z") and b not in bases:
            bases.append(b)

K = len(bases)
N = len(df)
jcp = np.empty((N, K, 3), dtype=float) 
for k, b in enumerate(bases):
    jcp[:, k, 0] = df[f"{b}_x"].to_numpy(dtype=float)/1000 # to set in m
    jcp[:, k, 1] = df[f"{b}_y"].to_numpy(dtype=float)/1000
    jcp[:, k, 2] = df[f"{b}_z"].to_numpy(dtype=float) /1000
 
 
# position of the cameras in the world frame (please add a function for that)
transformation_file = f"{base_path}/config/cam_params/{subject}/calib_mocap_2_cam0/soder.txt"
R_0, d_0, _,_ = load_transformation(transformation_file)
T_c0 = np.eye(4)
T_c0[:3, :3] = R_0
T_c0[:3, 3]  = d_0.reshape(3)


transformation_file = f"{base_path}/config/cam_params/{subject}/calib_mocap_2_cam2/soder.txt"
R_2, d_2, _,_ = load_transformation(transformation_file)
T_c2= np.eye(4)
T_c2[:3, :3] = R_2
T_c2[:3, 3]  = d_2.reshape(3)

transformation_file = f"{base_path}/config/cam_params/{subject}/calib_mocap_2_cam4/soder.txt"
R_4, d_4, _,_ = load_transformation(transformation_file)
T_c4 = np.eye(4)
T_c4[:3, :3] = R_4
T_c4[:3, 3]  = d_4.reshape(3)

transformation_file = f"{base_path}/config/cam_params/{subject}/calib_mocap_2_cam6/soder.txt"
R_6, d_6, _,_ = load_transformation(transformation_file)
T_c6 = np.eye(4)
T_c6[:3, :3] = R_6
T_c6[:3, 3]  = d_6.reshape(3)

# animate models
 
step=5
i0=0#350 # initial elaboration/display sample


print("meshcat animation...")
    
quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs()#set the human model uprigth
viz = pin.visualize.MeshcatVisualizer(model_mocap, collision_model_mocap, visual_model_mocap)
    
    
# one shared MeshCat viewer
viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
# mocap/reference model
viz_ref = MeshcatVisualizer(model_mocap, collision_model_mocap, visual_model_mocap)
viz_ref.initViewer(viewer)
viz_ref.loadViewerModel(rootNodeName="ref")
q = pin.neutral(model_mocap)

viz_ref.display(q)

# panda model
viz_robot = MeshcatVisualizer(model_robot, collision_model_robot, visual_model_robot)
viz_robot.initViewer(viewer)
viz_robot.loadViewerModel(rootNodeName="panda")
viz_robot.viewer["panda"].set_transform(T.homogeneous)
viz_robot.display(robot.q0)


# change background color
# native_viz = viz_ref.viewer
# native_viz["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
# native_viz["/Background"].set_property("bottom_color", [1, 1, 1])  # Same color → flat background
# grid_height = -0.0  # Negative z = lower the grid
# native_viz["/Grid"].set_transform(
# np.array([
#     [1, 0, 0, 0],  # Rotation (identity)
#     [0, 1, 0, 0],
#     [0, 0, 1, grid_height],
#     [0, 0, 0, 1]
# ]))

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
for i in range(i0,len(q_ref),step):#

    for j  in range(jcp.shape[1]):
        sphere_name = f'jcp_mocap{j}'
        
        addViewerSphere(viz_ref, sphere_name, 0.025, [0.5, 0.5, 0.5, 1])
        applyViewerConfiguration(viz_ref, sphere_name, np.hstack((jcp[i,j,:], np.array([0, 0, 0, 1]))))
    
    viz_ref.display(q_ref[i,:])
    
    # input()