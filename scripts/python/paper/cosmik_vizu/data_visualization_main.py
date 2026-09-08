import os
import pandas as pd
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
 
import yaml
import time 

from utils.vizutils import addViewerSphere, display_model_frames, applyViewerConfiguration
from utils.reader_parameters import parse_params, convert_to_class

from utils.model_utils_motif import Robot, build_biomechanical_model 

from utils.vizutils import display_model_frames,display_com, compute_cop, display_force, addViewerBox, addViewerSphere, applyViewerConfiguration

import matplotlib.pyplot as plt
 
from example_robot_data import load 


def set_tf(viz, name, T_world_obj):
   
        viz.viewer[name].set_transform(T_world_obj)
        
def draw_table(viz, T_world_table):
    """
    Build a simple table (top + 4 legs) and place it at T_world_table.
    Uses addViewerBox() for both MeshCat and Gepetto viewers.

    Parameters
    ----------
    viz : pin.visualize.MeshcatVisualizer or GepettoVisualizer
    T_world_table : (4,4) ndarray homogeneous transform (world_T_table)
    """
    # ---- table parameters (meters) ----
    L, W, T = .90, 1.80, 0.04      # tabletop length, width, thickness
    H       = 0.95                  # total height
    LEG     = 0.05                  # leg square cross-section
    INSET   = 0.05                  # leg inset from edges

    # ---- names (unique in the scene) ----
    top_name = "table_top"
    leg_names = [
        "table_leg_00", "table_leg_01",
        "table_leg_10", "table_leg_11",
    ]

    # ---- create geometries (via your helper) ----
    # tabletop (brown)
    addViewerBox(viz, top_name, L, W, T, rgba=[0.80, 0.60, 0.40, 1.0])
    # legs (slightly darker)
    for n in leg_names:
        addViewerBox(viz, n, LEG, LEG, H - T, rgba=[0.45, 0.45, 0.45, 1.0])

    # ---- local poses (relative to the table frame) ----
    def homog(R=np.eye(3), t=(0, 0, 0)):
        Tm = np.eye(4)
        Tm[:3, :3] = R
        Tm[:3,  3] = np.array(t, dtype=float)
        return Tm

    # tabletop center is at z = H - T/2
    T_local_top = homog(t=(0.0, 0.0, H - T/2))

    # leg centers
    xs = [+L/2 - INSET - LEG/2, -L/2 + INSET + LEG/2]
    ys = [+W/2 - INSET - LEG/2, -W/2 + INSET + LEG/2]
    z_leg = (H - T)/2
    T_local_legs = [
        homog(t=(xs[0], ys[0], z_leg)),
        homog(t=(xs[0], ys[1], z_leg)),
        homog(t=(xs[1], ys[0], z_leg)),
        homog(t=(xs[1], ys[1], z_leg)),
    ]

    # ---- apply world transforms ----
    set_tf(viz, top_name, T_world_table @ T_local_top)
    for n, Tl in zip(leg_names, T_local_legs):
        set_tf(viz, n, T_world_table @ Tl)

  

 
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))



# load parameters class defined in parameters.toml file
dict_param = parse_params(os.path.join(script_directory, 'parameters.toml'))
param = convert_to_class(dict_param) # define class param 


#pprint.pprint(param.groups_joint_torques)
###### Setup model (state the dof of interest from the general 37dof human model described in the documentation using active_joints)

 
  
 
#Load general human model 
urdf_name =  "human.urdf"#f"generated_{gender}_mass{weigth}_size{size}.urdf"

urdf_path = os.path.join(parent_directory,"Research/cosmik_visualization/model/human_urdf/urdf/", urdf_name)
print(urdf_path)
urdf_meshes_path = os.path.join(parent_directory, "Research/cosmik_visualization/model")

robot=Robot(urdf_path,urdf_meshes_path, param.free_flyer) 
model, collision_model,  visual_model, param=build_biomechanical_model(robot, param) 
data = model.createData()


model_mocap, collision_model_mocap,  visual_model_mocap, param=build_biomechanical_model(robot, param) 
data_mocap = model.createData()

# --- make visual_model_mocap grey ---
for go in visual_model_mocap.geometryObjects:
    go.overrideMaterial = True                     # ignore URDF materials/textures
    go.meshColor = np.array([0.4, 0.4, 0.4, 0.9])  # RGBA in [0,1]; e.g., mid-grey

# load panda model 

robot = load("panda")
model_robot = robot.model

visual_model_robot = robot.visual_model
collision_model_robot = robot.collision_model
data_robot = robot.data


# model=robot.model
# visual_model = robot.visual_model
# collision_model = robot.collision_model
# data = model.createData()


#Load data model
csv_path = "data_for_vid_cosmik/q_mocap.csv"  # your file
q_ref = pd.read_csv(csv_path).to_numpy(dtype=float)  # shape: (num_samples, num_joints_including_FF)

csv_path = "data_for_vid_cosmik/q_cosmik_swika.csv"  # your file
q_est = pd.read_csv(csv_path).to_numpy(dtype=float)  # shape: (num_samples, num_joints_including_FF)

csv_path = "data_for_vid_cosmik/robot_data_interpolated.csv"  # your file
df = pd.read_csv(csv_path, parse_dates=["_cam_time", "timestamp"])

# determine the synchro between the robot and the cam data
t_cam   = pd.read_csv("data_for_vid_cosmik/camera_0_timestamps.csv",   parse_dates=["timestamp"])
t_robot = pd.read_csv(csv_path, parse_dates=["timestamp"])


def to_utc(s):
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")

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
with open("data_for_vid_cosmik/robot_base_pose.yaml") as f:
    Y = yaml.safe_load(f)["world_T_robot"]
R = np.array(Y["rotation_matrix"], dtype=float)
t = np.array(Y["translation"], dtype=float)
T = pin.SE3(R, t)


# load triangualated filtered  keypoints
csv_path = "data_for_vid_cosmik/3d_keypoints_filtered.csv" 

 
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
keypoints_filt = np.empty((N, K, 3), dtype=float)

for k, b in enumerate(bases):
    keypoints_filt[:, k, 0] = df[f"{b}_x"].to_numpy(dtype=float)
    keypoints_filt[:, k, 1] = df[f"{b}_y"].to_numpy(dtype=float)
    keypoints_filt[:, k, 2] = df[f"{b}_z"].to_numpy(dtype=float)

# load trianulated non filtred keypoints
csv_path = "data_for_vid_cosmik/3d_keypoints.csv"   

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
keypoints = np.empty((N, K, 3), dtype=float)

for k, b in enumerate(bases):
    keypoints[:, k, 0] = df[f"{b}_x"].to_numpy(dtype=float)
    keypoints[:, k, 1] = df[f"{b}_y"].to_numpy(dtype=float)
    keypoints[:, k, 2] = df[f"{b}_z"].to_numpy(dtype=float)



# load covariances

df0 = pd.read_csv("data_for_vid_cosmik/robot_welding_pose_scores_0.csv")
df1 = pd.read_csv("data_for_vid_cosmik/robot_welding_pose_scores_2.csv")

# rename score cols to keep them distinct, then merge on 'frame'
scores0 = [c for c in df0.columns if c.endswith("_score")]
scores1 = [c for c in df1.columns if c.endswith("_score")]
df0_ren = df0.rename(columns={c: f"{c}_c0" for c in scores0})
df1_ren = df1.rename(columns={c: f"{c}_c1" for c in scores1})

df_merged = pd.merge(df0_ren[["frame"] + [f"{c}_c0" for c in scores0]],
                     df1_ren[["frame"] + [f"{c}_c1" for c in scores1]],
                     on="frame", how="inner")

# example: arrays per camera (aligned by frame)
scores_cam0 = df_merged[[f"{c}_c0" for c in scores0]].to_numpy(float)
scores_cam1 = df_merged[[f"{c}_c1" for c in scores1]].to_numpy(float)


def rolling_var(sig, W):
    """Return rolling variance of 1D array sig with window W (NaN for first W-1)."""
    N = len(sig)
    out = np.full(N, np.nan)
    for t in range(W-1, N):
        seg = sig[t-W+1:t+1]
        out[t] = np.nanvar(seg, ddof=1)
    return out

names = list(IDX.keys())
K = len(names)
N = keypoints.shape[0]
cols = 5
rows = math.ceil(K / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.2), sharex=True)
axes = axes.ravel()

for i, name in enumerate(names):
    j = IDX[name]
    x = keypoints[:, j, 0]
    y = keypoints[:, j, 1]
    z = keypoints[:, j, 2] if keypoints.shape[2] > 2 else None

    vx = rolling_var(x, W)
    vy = rolling_var(y, W)
    ax = axes[i]
    ax.plot(vx, label='var(x)', linewidth=1)
    ax.plot(vy, label='var(y)', linewidth=1)
    if z is not None:
        vz = rolling_var(z, W)
        ax.plot(vz, label='var(z)', linewidth=1)

    ax.set_title(name)
    ax.grid(True, alpha=0.3)

# hide any unused subplots
for k in range(i+1, len(axes)):
    axes[k].axis('off')

axes[0].legend(loc='upper right', fontsize=8)
fig.suptitle(f"Rolling covariance (variance) per marker, window={W} frames", y=0.995)
plt.tight_layout()
plt.show()




# Filter keypoints using kalman filter 

# Indices for your 26-joint layout (match your CSV order)
IDX = {
    "Nose":0,"LEye":1,"REye":2,"LEar":3,"REar":4,
    "LShoulder":5,"RShoulder":6,"LElbow":7,"RElbow":8,"LWrist":9,"RWrist":10,
    "LHip":11,"RHip":12,"LKnee":13,"RKnee":14,"LAnkle":15,"RAnkle":16,
    "Head":17,"Neck":18,"midHip":19,
    "LBigToe":20,"RBigToe":21,"LSmallToe":22,"RSmallToe":23,"LHeel":24,"RHeel":25,
}

# A light set of "bones" (pairs of indices) to keep lengths roughly constant
BONES = [
    (IDX["Head"], IDX["Neck"]),
    (IDX["Neck"], IDX["midHip"]),
    (IDX["Neck"], IDX["LShoulder"]), (IDX["LShoulder"], IDX["LElbow"]), (IDX["LElbow"], IDX["LWrist"]),
    (IDX["Neck"], IDX["RShoulder"]), (IDX["RShoulder"], IDX["RElbow"]), (IDX["RElbow"], IDX["RWrist"]),
    (IDX["midHip"], IDX["LHip"]), (IDX["LHip"], IDX["LKnee"]), (IDX["LKnee"], IDX["LAnkle"]),
    (IDX["midHip"], IDX["RHip"]), (IDX["RHip"], IDX["RKnee"]), (IDX["RKnee"], IDX["RAnkle"]),
    (IDX["LAnkle"], IDX["LHeel"]), (IDX["RAnkle"], IDX["RHeel"]),
    (IDX["LAnkle"], IDX["LBigToe"]), (IDX["RAnkle"], IDX["RBigToe"]),
]


 

def init_kf_state(K, dt=1/30, q=5e-3, r=5e-4, gate_sigma=4.0,
                  bones=None, ref_len=None, bone_alpha=0.5, n_proj_iters=2,
                  z0=None):
    """
    Initialize state for per-joint constant-velocity KF (+ soft bone projection).
    K          : number of joints
    bones      : list of (i,j) index pairs; None to disable projection
    ref_len    : array of target lengths (len(bones),); None -> projection disabled
    z0         : optional (K,3) initial positions
    """
    I3 = np.eye(3); dt = float(dt)
    F  = np.block([[I3, dt*I3],
                   [np.zeros((3,3)), I3]])                 # 6x6
    H  = np.block([I3, np.zeros((3,3))])                  # 3x6
    Q  = q * np.block([[(dt**3/3)*I3, (dt**2/2)*I3],
                       [(dt**2/2)*I3,  dt*I3]])
    R  = r * I3

    X = np.zeros((K, 6), dtype=float)   # [pos(3), vel(3)]
    P = np.tile(np.eye(6), (K,1,1)).astype(float)
    if z0 is not None:
        z0 = np.asarray(z0, float)
        m = ~np.isnan(z0).any(axis=1)
        X[m, :3] = z0[m]

    state = dict(
        K=K, F=F, H=H, Q=Q, R=R, gate2=float(gate_sigma**2),
        X=X, P=P, bone_alpha=float(bone_alpha), n_proj_iters=int(n_proj_iters)
    )
    if bones is not None and ref_len is not None:
        bones = np.asarray(bones, int)
        state["I_idx"] = bones[:,0]
        state["J_idx"] = bones[:,1]
        state["ref_len"] = np.asarray(ref_len, float)
    else:
        state["I_idx"] = state["J_idx"] = state["ref_len"] = None
    return state

def kf_keypoints_step(z, state):
    """
    One update step. Input z: (K,3) noisy positions (NaNs allowed).
    Returns filtered positions (K,3) and updates `state` in-place.
    """
    z = np.asarray(z, float)
    K = state["K"]; F=state["F"]; H=state["H"]; Q=state["Q"]; R=state["R"]; gate2=state["gate2"]
    X=state["X"]; P=state["P"]

    # predict
    for j in range(K):
        X[j] = F @ X[j]
        P[j] = F @ P[j] @ F.T + Q

    # update (skip NaNs, Mahalanobis gating; use solve instead of inverse)
   
    I6 = np.eye(6)
    updated = np.zeros(K, dtype=bool)  # track who actually updated

    for j in range(K):
        meas = z[j]
        if np.any(np.isnan(meas)):
            continue
        
        Rj = state.get("Rj", R)[j]   # <— per-joint R if provided
        y  = meas - (H @ X[j])  # innovation (3,)
        S  = H @ P[j] @ H.T + Rj  # innovation cov (3,3)
            
        # y = meas - (H @ X[j])              # innovation (3,)
        # S = H @ P[j] @ H.T + R             # innovation cov (3,3)

        # Mahalanobis gate
        try:
            v = np.linalg.solve(S, y)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(S, y, rcond=None)[0]
        md2 = float(y @ v)
        if md2 > gate2:
            # optional: debug drift for joint 9
            if j == 9:
                print(f"[KF] j=9 gated: md2={md2:.2f} > {gate2:.2f}")
            continue

        # Kalman gain: K = P H^T S^{-1}
        HP = H @ P[j]                       # (3,6)
        try:
            S_inv_HP = np.linalg.solve(S, HP)
        except np.linalg.LinAlgError:
            S_inv_HP = np.linalg.lstsq(S, HP, rcond=None)[0]
        Kk = S_inv_HP.T                     # (6,3)

        # state update
        X[j] = X[j] + Kk @ y

        # Joseph form for P (keeps PSD & symmetry)
        KH  = Kk @ H                        # (6,6)
        Pj  = (I6 - KH) @ P[j] @ (I6 - KH).T + Kk @ R @ Kk.T
        P[j] = 0.5 * (Pj + Pj.T)            # re-symmetrize
        updated[j] = True

   

    Pnow = X[:, :3].copy()

    # soft bone-length projection (vectorized)
    I_idx, J_idx, ref_len = state["I_idx"], state["J_idx"], state["ref_len"]
    if (I_idx is not None) and (ref_len is not None) and state["n_proj_iters"] > 0 and state["bone_alpha"] > 0:
        B = len(I_idx)
        for _ in range(state["n_proj_iters"]):
            vi = Pnow[I_idx]                 # (B,3)
            vj = Pnow[J_idx]                 # (B,3)
            v  = vj - vi                     # (B,3)
            d  = np.linalg.norm(v, axis=1)   # (B,)
            u  = np.zeros_like(v)
            nz = d > 1e-12
            u[nz] = v[nz] / d[nz,None]
            scale = state["bone_alpha"] * 0.5 * (ref_len - d)   # (B,)
            delta = (scale[:,None]) * u
            np.add.at(Pnow, I_idx, -delta)
            np.add.at(Pnow, J_idx, +delta)
        # keep state consistent
        X[:, :3] = Pnow
 
        
        
        
    return Pnow


 
 
 
 

 
 
 
 
 































q0=pin.neutral(model) 
 
 
if param.viewer=="meshcat":

    print("meshcat animation...")
     
      # frames to be displayed
    #frame2display=["root_joint","left_hand"]
    frame2display=["middle_lumbar_Z","right_hand"]
    display_model_frames(model, visual_model, frame2display, param) 
    
    quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs()#set the human model uprigth
    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
       
      
    # one shared MeshCat viewer
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

    # estimated model
    viz_est = MeshcatVisualizer(model, collision_model, visual_model)
    viz_est.initViewer(viewer)
    viz_est.loadViewerModel(rootNodeName="est")     # older versions

    # mocap/reference model
    viz_ref = MeshcatVisualizer(model_mocap, collision_model_mocap, visual_model_mocap)
    viz_ref.initViewer(viewer)
    viz_ref.loadViewerModel(rootNodeName="ref")


    # panda model
    viz_robot = MeshcatVisualizer(model_robot, collision_model_robot, visual_model_robot)
    viz_robot.initViewer(viewer)
    viz_robot.loadViewerModel(rootNodeName="panda")
    viz_robot.viewer["panda"].set_transform(T.homogeneous)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.loadViewerModel()
    
    # viz_mocap = pin.visualize.MeshcatVisualizer(model_mocap, collision_model_mocap, visual_model_mocap) 
    # viz_mocap.initViewer(viz.viewer) # share the same viewer/session
    # viz_mocap.loadViewerModel()
    
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
     
    # #driver = webdriver.Safari()
    # #driver.get(viz.viewer.url())  # MeshCat URL
   
    
    # add forces plate   
    fp_names = ["forceplate1", "forceplate2","forceplate3", "forceplate4"]   
    # meters, (x, y) per force plate
    fp_dim = [
        (0.60, 1.20),  # FP1
        (0.60, 0.40),  # FP2
        (0.60, 0.80),  # FP3
        (0.9, 1.8),  # FP4
    ]
    fp_centers = [
    ( -2.0,  0.0, 0.0),# FP1
    ( -1.7,  0.0, 0.0),# FP2
    ( -1.0,  0.0, 0.0),# FP3
    ( 0.9,  0.45, 0.0),# FP4
    ]

    for j, ((sx, sy), (cx, cy, cz)) in enumerate(zip(fp_dim, fp_centers), start=1):
        name = f"force_plate_{j}"
        addViewerBox(viz_robot, name, sx, sy, 0.01, rgba=[0.5, 0.5, 0.5, 1.0]) # create a box

        # build world transform (centered box: put center at (cx,cy,cz))
        T = np.eye(4)
        T[:3, 3] = [cx, cy, cz + 0.01/2.0]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
        set_tf(viz_robot, name, T)
    
    # add wall and windows
    
    addViewerBox(viz_robot, "wall", 10, 0.5, 3, rgba=[0.96, 0.96, 0.86, 1.0]) # create a box
    T = np.eye(4)
    T[:3, 3] = [0,3, 1.5]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    set_tf(viz_robot,  "wall", T)
  
  
    addViewerBox(viz_robot, "window1", 2, 0.1, 1.5, rgba=[0.25, 0.25, 0.25, 1.0]) # create a box
    T = np.eye(4)
    T[:3, 3] = [-3,2.75, 1.8]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    set_tf(viz_robot,  "window1", T)
  
    addViewerBox(viz_robot, "window2", 2, 0.1, 1.5, rgba=[0.25, 0.25, 0.25, 1.0]) # create a box
    T = np.eye(4)
    T[:3, 3] = [3,2.75, 1.8]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    set_tf(viz_robot,  "window2", T)
    
    addViewerBox(viz_robot, "door", 1.2, 0.1, 2.2, rgba=[0.2, 0.2, 0.2, 1.0]) # create a box
    T = np.eye(4)
    T[:3, 3] = [0,2.75, 1.1]  # if your "floor" is at z=0 and you want bottom on floor, pass cz=0
    set_tf(viz_robot,  "door", T)
    
    

    step=1
    i0=500
    # jcp_noisy: (N, K, 3)
    z0 = keypoints[0,:,:]                # first frame, shape (K,3)

    bones = np.asarray(BONES, int)    # list of (i,j) pairs
    I, J = bones[:, 0], bones[:, 1]

    # distances for each bone in the first frame
    ref_len = np.zeros(len(bones), dtype=float)

    # valid pairs (no NaNs on either endpoint)
    valid = ~(np.isnan(z0[I]).any(axis=1) | np.isnan(z0[J]).any(axis=1))
    ref_len[valid] = np.linalg.norm(z0[I[valid]] - z0[J[valid]], axis=1)

    # optional: fill any missing with median of valid (or leave zeros)
    if not valid.all():
        m = np.median(ref_len[valid]) if np.any(valid) else 0.0
        ref_len[~valid] = m


    state = init_kf_state(26, dt=step/40,
                    bones=None,           # list of (i,j)
                    ref_len=None,       # length per bone
                    bone_alpha=0.6, n_proj_iters=3,z0=z0)
    
    # after you put R into state
    state["Rj"] = np.tile(state["R"][None, :, :], (state["K"], 1, 1))
    state["Rj"][9] *= 4.0   # 2–5x is typical; tune

    
    print(ref_len)
    
    wrist=[]
    wrist_filt=[]
    wrist_kf=[]
    for i in range(i0,i0+150,step):#len(q_ref),step):#
         
         
         
      
        # For each new sample z_t (K,3):
        keypoints_kf = kf_keypoints_step(keypoints[i,:,:], state)   # (K,3)

        wrist_filt.append(keypoints_filt[i,9,:])
        wrist_kf.append(keypoints_kf[9,:])
        wrist.append(keypoints[i,9,:])
         
        
        for j  in range(keypoints.shape[1]):
            sphere_name = f'keypoints_kf{j}'
            addViewerSphere(viz_est, sphere_name, 0.025, [0, 1, 0, 1])
            applyViewerConfiguration(viz_est, sphere_name, np.hstack((keypoints_kf[j,:], np.array([0, 0, 0, 1]))))
     
    
        for j  in range(keypoints.shape[1]):
            sphere_name = f'keypoints_{j}'
            addViewerSphere(viz_est, sphere_name, 0.025, [0, 0, 1, 1])
            applyViewerConfiguration(viz_est, sphere_name, np.hstack((keypoints[i,j,:], np.array([0, 0, 0, 1]))))
    
        if i>=first["cam_idx"]:
             viz_robot.display(q_robot[i-first["cam_idx"],:])
 
        viz_est.display(q_est[i,:])
        viz_ref.display(q_ref[i,:])
       
        T_world_table = np.eye(4)
        T_world_table[:3, 3] = [0.9, -0.6, 0.0]

        draw_table(viz_robot, T_world_table) 
        #input()
    #     # Save screenshot
    #     #save_img=driver.save_screenshot(f"example/squat/video/frame_{i:04d}.png")
    #     #print(save_img)
        #time.sleep(0.01)
       
       
    #driver.quit()
    
    
    # command line to type in the terminal to save a video
    # ffmpeg -framerate 40 -start_number 2 -i example/squat/video/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 15 example/squat/video/squat.mp4


    plt.plot(wrist)
    plt.plot(wrist_filt,'k--')
    plt.plot(wrist_kf,'--')
    
    plt.show()
