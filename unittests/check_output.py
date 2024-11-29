import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

q = pd.read_csv(os.path.join(rt_cosmik_path,'output/q.csv'))
augmented_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/augmented_markers_positions.csv'))
keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))

# check data timestamps
# Ensure the 'Time' column is in datetime format
q["Time"] = pd.to_datetime(q["Time"])

# Calculate differences in seconds
diff_t = q["Time"].diff().dt.total_seconds()

# Compute the mean of the differences
mean_diff = diff_t.mean()
print(mean_diff)

q_to_plot = q.to_numpy()[:,2:]

dt = mean_diff
T = q_to_plot.shape[0]*dt

t = np.arange(0,T,dt)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2-row, 3-column layout

# Flatten the axes array for easy indexing
axs = axs.flatten()

for i in range(0, q_to_plot.shape[1] + 1):
    if i == 0:
        # Add the diff_t plot in the first subplot
        axs[i].plot(range(len(diff_t)), diff_t, label="Time Differences (s)", color="blue")
        axs[i].axhline(mean_diff, color="red", linestyle="--", label=f"Mean: {mean_diff:.6f}s")
        axs[i].set_ylabel("Time Diff [s]")
        axs[i].set_xlabel("Sample Index")  # Independent x-axis for diff_t
        axs[i].legend()
    else:
        # Plot other data in subsequent subplots
        axs[i].plot(t, q_to_plot[:, i - 1])
        axs[i].set_ylabel(f"q_{i-1} [rad]")
        axs[i].set_xlabel("Time [s]")
        axs[i].legend()
    
    # Keep only left and bottom frame lines
    axs[i].spines["top"].set_visible(False)
    axs[i].spines["right"].set_visible(False)
    
    # Show only two y-axis ticks (min and max)
    ymin, ymax = axs[i].get_ylim()
    axs[i].set_yticks([ymin, ymax])

# Add a title to the entire figure
fig.suptitle("RT COSMIK Results", fontsize=16)

# Hide unused subplots if there are any
for j in range(q_to_plot.shape[1] + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.show()
