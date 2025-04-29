import pandas as pd
import numpy as np
from src.rtcosmik.utils.read_write_utils import read_mks_data, udp_csv_to_dataframe,marker_data_to_dataframe
from src.rtcosmik.config_loader import settings
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

no_trial = "trial3"
task = "sit_to stand"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data.csv"
output_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_data_filled.csv"
mks_names = settings.marker_mocap_names_tech

def fill_gaps_with_cubic_spline(df, time_col=None):
    df_interp = df.replace(0.0, np.nan)
    if time_col:
        x = df[time_col].values
        df_interp = df.drop(columns=[time_col])
    else:
        x = np.arange(len(df))

    for col in df_interp.columns:
        y = df_interp[col].values
        mask = ~np.isnan(y)
        if mask.sum() >= 4:
            cs = CubicSpline(x[mask], y[mask])
            df_interp[col] = cs(x)
    if time_col:
        df_interp[time_col] = x
    return df_interp


def plot_marker_trajectories(df_wide, marker_names, filled_df):
    """
    For each marker, plot 6 subplots in a figure:
    - 3 subplots for original x, y, z
    - 3 subplots for filled x, y, z

    Parameters:
        df_wide (pd.DataFrame): Original marker data with columns marker_x, marker_y, marker_z.
        marker_names (list): List of marker base names.
        filled_df (pd.DataFrame): Interpolated marker data.
    """
    for marker in marker_names:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        axes_labels = ['x', 'y', 'z']

        for i, axis in enumerate(axes_labels):
            col = f"{marker}_{axis}"
            # Original
            axes[i, 0].plot(df_wide[col].index, df_wide[col], color='b', label=f'Original {axis}')
            missing_idx = df_wide[col][df_wide[col] == 0.0].index
            axes[i, 0].plot(missing_idx, df_wide[col].loc[missing_idx], 'ro', label='Missing (== 0)')
            axes[i, 0].set_ylabel(f"{axis}")
            axes[i, 0].legend()
            axes[i, 0].grid(True)

            # Filled
            axes[i, 1].plot(filled_df.index, filled_df[col],color='r',label=f'Filled {axis}')
            axes[i, 1].legend()
            axes[i, 1].grid(True)

        for ax in axes[-1]:
            ax.set_xlabel("Frame")
        fig.suptitle(f"Marker: {marker}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



df_wide = udp_csv_to_dataframe(path_to_csv, mks_names)

# Gap filling step
df_wide_filled = fill_gaps_with_cubic_spline(df_wide)
plot_marker_trajectories(df_wide, mks_names,filled_df=df_wide_filled)

mks_cols = []
for name in mks_names:
    mks_cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

# Save using the correct column order
df_wide_filled[mks_cols].to_csv(output_path, index=False)