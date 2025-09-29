import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import numpy as np
import pandas as pd 
import sys
from src.rtcosmik.utils.read_write_utils  import read_mks_data


df = pd.read_csv("output/force_plate/static_trajectories.csv",delimiter=',')

frames = df["Frame"] if "Frame" in df.columns else range(len(df))
mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))

mks_dict, start_sample_dict = read_mks_data(df, start_sample=0) #convert to m if needed 
markers = mks_dict[0]

Sensix2_9D = np.array([5.77,31.37,-330.83,57853.5,472.349,-3936.01,903.428,-1066.88,0])
Sensix3_9D = np.array([12.10,-22.18,-321.644,-49639.1,6841.17,603.452,923.269,-1377.67,0])


# === Initialiser le visualiseur Gepetto ===
viz = GepettoVisualizer()
try:
    viz.initViewer()
except ImportError as err:
    print("Install gepetto-viewer.")
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Start gepetto-viewer before running this script.")
    sys.exit(0)

viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
viz.viewer.gui.addXYZaxis('world/sensix2_frame', [255, 0., 0, 1.], 0.04, 0.2)
viz.viewer.gui.addXYZaxis('world/sensix3_frame', [255, 0., 0, 1.], 0.04, 0.2)

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

R_fp = pin.utils.rotate('x',-np.pi)@pin.utils.rotate('z',np.pi/2)

M_sensix2 = pin.SE3(R_fp, np.array([0.902,-0.892,0.0]))
M_sensix3 = pin.SE3(R_fp, np.array([0.902,-1.532,0.0]))

place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.zeros((3,1))))
place(viz, 'world/sensix2_frame', M_sensix2)
place(viz, 'world/sensix3_frame', M_sensix3)

# === Ajouter les sphères ===
for name in mks_names:
    if name == "L_lwrist_study" or name == "r_lwrist_study" or name == "r_knee_study" or name == "L_knee_study" or name == "r_ankle_study" or name == "L_ankle_study" or name == "r_lelbow_study"or name == "L_lelbow_study" or name == 'r_5meta_study' or name == 'L_5meta_study':
        viz.viewer.gui.addSphere('world/'+name,0.01,[1,0,0,1])
    
    if name == "L.PSIS_study" or name == "r.PSIS_study":
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,1,0,1])
    else :
        viz.viewer.gui.addSphere('world/'+name,0.01,[0,0,1,1])

# === Visualiser frame par frame ===
for i, frame in enumerate(mks_dict):
    if i ==0:
        for name in mks_names:
            pos = frame[name].reshape(3,)/1000
            place(viz, f'world/{name}', pin.SE3(np.eye(3), pos.reshape(3,1)))

pin_force_sensix2 = pin.Force(np.array([5.77,31.37,-330.83,57853.5/1000,472.349/1000,-3936.01/1000])).se3ActionInverse(M_sensix2)
pin_force_sensix3 = pin.Force(np.array([12.10,-22.18,-321.644,-49639.1/1000,6841.17/1000,603.452/1000])).se3ActionInverse(M_sensix3)

COP_sensix2 = np.array([pin_force_sensix2.angular[0]/pin_force_sensix2.linear[2], -pin_force_sensix2.angular[1]/pin_force_sensix2.linear[2],0])
COP_sensix3 = np.array([pin_force_sensix3.angular[0]/pin_force_sensix3.linear[2], -pin_force_sensix3.angular[1]/pin_force_sensix3.linear[2],0])

viz.viewer.gui.addSphere('world/COP_sensix2',0.02,[1,1,0,1])
viz.viewer.gui.addSphere('world/COP_sensix3',0.02,[1,1,0,1])

place(viz, 'world/COP_sensix2', pin.SE3(np.eye(3), COP_sensix2.reshape(3,1)))
place(viz, 'world/COP_sensix3', pin.SE3(np.eye(3), COP_sensix3.reshape(3,1)))

viz.viewer.gui.addSphere('world/COP_sensix2_meas',0.02,[0,1,1,1])
viz.viewer.gui.addSphere('world/COP_sensix3_meas',0.02,[0,1,1,1])

place(viz, 'world/COP_sensix2_meas', pin.SE3(np.eye(3), np.array([903.428/1000,-1066.88/1000,0]).reshape(3,1)))
place(viz, 'world/COP_sensix3_meas', pin.SE3(np.eye(3), np.array([923.269/1000,-1377.67/1000,0]).reshape(3,1)))

right_foot_calc = mks_dict[0]['r_calc_study'].reshape(3,)/1000
right_foot_toe = mks_dict[0]['r_toe_study'].reshape(3,)/1000
left_foot_calc = mks_dict[0]['L_calc_study'].reshape(3,)/1000
left_foot_toe = mks_dict[0]['L_toe_study'].reshape(3,)/1000

right_foot_length = np.linalg.norm(right_foot_toe - right_foot_calc)
right_foot_vector = (right_foot_toe - right_foot_calc) / right_foot_length if right_foot_length > 0 else np.array([1,0,0])

COP_right_est = right_foot_calc + right_foot_vector * (right_foot_length*0.40)

right_foot_length = np.linalg.norm(right_foot_toe - right_foot_calc)
right_foot_vector = (right_foot_toe - right_foot_calc) / right_foot_length if right_foot_length > 0 else np.array([1,0,0])

COP_right_est = right_foot_calc + right_foot_vector * (right_foot_length*0.40)

left_foot_length = np.linalg.norm(left_foot_toe - left_foot_calc)
left_foot_vector = (left_foot_toe - left_foot_calc) / left_foot_length if left_foot_length > 0 else np.array([1,0,0])

COP_left_est = left_foot_calc + left_foot_vector * (left_foot_length*0.40)

viz.viewer.gui.addSphere('world/COP_lfoot_est',0.02,[0,1,1,1])
viz.viewer.gui.addSphere('world/COP_rfoot_est',0.02,[0,1,1,1])

place(viz, 'world/COP_lfoot_est', pin.SE3(np.eye(3), COP_left_est.reshape(3,1)))
place(viz, 'world/COP_rfoot_est', pin.SE3(np.eye(3), COP_right_est.reshape(3,1)))

M_COP_sensix2_meas = pin.SE3(np.eye(3), np.array([903.428/1000,-1066.88/1000,0]).reshape(3,1))
M_COP_sensix3_meas = pin.SE3(np.eye(3), np.array([923.269/1000,-1377.67/1000,0]).reshape(3,1))

M_COP_sensix2_est = pin.SE3(np.eye(3), COP_left_est.reshape(3,1))
M_COP_sensix3_est = pin.SE3(np.eye(3), COP_right_est.reshape(3,1))

def solve_rigid_2d(A, B):
    """
    A, B: (N,2) point sets (N>=2). Return (R2, t2) with R2 in SO(2), t2 in R^2,
    minimizing ||R2*A_i + t2 - B_i||.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:   # enforce proper rotation (no reflection)
        Vt[1, :] *= -1
        R2 = Vt.T @ U.T
    t2 = cb - R2 @ ca
    return R2, t2

def rms(err_rows): 
    e = np.asarray(err_rows, float)
    return float(np.sqrt(np.mean(np.sum(e*e, axis=1))))

# ---- Build correspondence (RIGHT->Sensix2, LEFT->Sensix3)
cop_est_R = COP_right_est[:2]
cop_est_L = COP_left_est [:2]
A = np.vstack([cop_est_R, cop_est_L])   # estimated (markers)

cop_meas_R = np.array([903.428/1000,  -1066.88/1000])  # sensix2 (right)
cop_meas_L = np.array([923.269/1000,  -1377.67/1000])  # sensix3 (left)
B = np.vstack([cop_meas_R, cop_meas_L])                # measured (plates)

# ---- Solve transform
R2, t2 = solve_rigid_2d(A, B)
R = np.eye(3); R[:2, :2] = R2
t = np.array([t2[0], t2[1], 0.0])
T_marker_to_plate = pin.SE3(R, t)

print(f"[Align] yaw (deg): {np.degrees(np.arctan2(R2[1,0], R2[0,0])):.3f}  "
      f"tx,ty (m): {t2[0]:.4f}, {t2[1]:.4f}")
print(f"[Align] RMS before: {rms(A - B):.4f} m  "
      f"after: {rms((A @ R2.T + t2) - B):.4f} m")

# ---- Helpers
def apply_T_point(T, p3):
    p3 = np.asarray(p3).reshape(3,)
    return T.rotation @ p3 + T.translation

# ---- Re-place all markers in aligned world (using your 'frame' from i==0)
for name in mks_names:
    p = frame[name].reshape(3,) / 1000.0
    p_aligned = apply_T_point(T_marker_to_plate, p)
    place(viz, f'world/{name}', pin.SE3(np.eye(3), p_aligned.reshape(3,1)))

# # ---- Update estimated CoPs after alignment (measured stay as-is)
# COP_left_est_aligned  = apply_T_point(T_marker_to_plate, COP_left_est)
# COP_right_est_aligned = apply_T_point(T_marker_to_plate, COP_right_est)
# place(viz, 'world/COP_lfoot_est', pin.SE3(np.eye(3), COP_left_est_aligned.reshape(3,1)))
# place(viz, 'world/COP_rfoot_est', pin.SE3(np.eye(3), COP_right_est_aligned.reshape(3,1)))

# # (Optional) move the world/base_frame gizmo to the new aligned origin
# place(viz, 'world/base_frame', T_marker_to_plate)

