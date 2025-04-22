import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe


no_trial = "trial_2"
task = "trial_upper_qp"
path_to_csv = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_pose.csv"
df = pd.read_csv(path_to_csv) #,nrows=4000)
mks_names = settings.marker_mocap_names
if 'marker_data' in df.columns:
    df = marker_data_to_dataframe(df, mks_names)

result_markers1, _ = read_mks_data(df)

def plot_selected_markers(marker_data, markers_to_plot=None):
    """
    Plot X, Y, Z values over time for selected markers in separate figures,
    with X, Y, and Z in subplots for each marker.

    Parameters:
        marker_data (list of dict): Each element is a dict {marker_name: np.array([x, y, z])}
        markers_to_plot (list of str): List of marker names to plot. If None, plots all markers.
    """
    # If no specific markers are selected, plot all markers
    if markers_to_plot is None:
        markers_to_plot = marker_data[0].keys()

    # Convert to numpy arrays for each marker
    for marker in markers_to_plot:
        # Check if the marker exists in the data
        if marker not in marker_data[0]:
            print(f"Marker '{marker}' not found in the data.")
            continue

        # Extract X, Y, Z values for the marker
        x_vals = [frame[marker][0] for frame in marker_data]
        y_vals = [frame[marker][1] for frame in marker_data]
        z_vals = [frame[marker][2] for frame in marker_data]
        frames = np.arange(len(marker_data))

        # Create a figure with 3 subplots (for X, Y, Z)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plotting X, Y, Z in separate subplots
        axs[0].plot(frames, x_vals, color='r')
        axs[0].set_title(f"{marker} - X")
        axs[0].set_ylabel("X")
        axs[0].grid(True)

        axs[1].plot(frames, y_vals, color='g')
        axs[1].set_title(f"{marker} - Y")
        axs[1].set_ylabel("Y")
        axs[1].grid(True)

        axs[2].plot(frames, z_vals, color='b')
        axs[2].set_title(f"{marker} - Z")
        axs[2].set_ylabel("Z")
        axs[2].set_xlabel("Frame")
        axs[2].grid(True)

        # Adjust layout and show
        plt.tight_layout()
        plt.show()

plot_selected_markers(result_markers1)