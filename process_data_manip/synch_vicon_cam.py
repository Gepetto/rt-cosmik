#plot q_cosmik and q_mocap to check if i have same pattern
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings

no_trial = "trial_2"
task = "trial_lower"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_mocap_ipopt.csv"
path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/joint_angles.csv"
output_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mocap_on_cosmik_frames.csv"


mocap_rate  = 100.0   # Hz
cosmik_rate =  40.0   # Hz

# --- 1. Load Cosmik data and compute relative times ---
df_cos       = pd.read_csv(path_cosmik)
frame_nums   = df_cos['Frame_0'].values
first_frame  = frame_nums[0]           # e.g. 311 if that's your first saved frame
t_cos        = (frame_nums - first_frame) / cosmik_rate
print(t_cos)
# --- 2. Load Mocap data ---
df_moc       = pd.read_csv(path_mocap)
N_moc        = len(df_moc)

# --- 3. Compute fractional mocap indices ---
x    = t_cos * mocap_rate
i0   = np.floor(x).astype(int)
i1   = np.ceil(x).astype(int)

# Clamp both indices so they never step out of bounds
i0 = np.clip(i0, 0, N_moc - 1)
i1 = np.clip(i1, 0, N_moc - 1)

# Interpolation weights
alpha = x - i0

# --- 4. Interpolate each channel ---
out = {}
for col in df_moc.columns:
    y = df_moc[col].values
    y0 = y[i0]
    y1 = y[i1]
    out[col] = y0 * (1 - alpha) + y1 * alpha

# --- 5. Save result (no time column) ---
df_out = pd.DataFrame(out)
df_out.to_csv(output_path, index=False)

print(f"Saved interpolated mocap → cosmik (frames) to:\n  {output_path}")