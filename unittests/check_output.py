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
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from utils.model_utils import Robot, get_jcp_global_pos
from utils.viz_utils import place

q = pd.read_csv(os.path.join(rt_cosmik_path,'output/q.csv'))
data_markers = pd.read_csv(os.path.join(rt_cosmik_path,'output/augmented_markers_positions.csv'))
data_keypoints = pd.read_csv(os.path.join(rt_cosmik_path,'output/keypoints_3d_positions.csv'))

result_markers = []
for frame, group in data_markers.groupby("Frame"):
    frame_dict = {row["Marker"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    result_markers.append(frame_dict)

result_keypoints = []
for frame, group in data_keypoints.groupby("Frame"):
    frame_dict = {row["Keypoint"]: np.array([row["X"], row["Y"], row["Z"]]) for _, row in group.iterrows()}
    if "Hip" in frame_dict:
        frame_dict["midHip"] = frame_dict.pop("Hip")
    result_keypoints.append(frame_dict)

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

human = Robot(os.path.join(rt_cosmik_path,'urdf/human_5dof.urdf'),rt_cosmik_path) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

pin.framesForwardKinematics(human_model,human_data, pin.neutral(human_model))
pos_ankle_calib = human_data.oMi[human_model.getJointId('ankle_Z')].translation

# VISUALIZATION

viz = GepettoVisualizer(human_model,human_collision_model,human_visual_model)
try:
    viz.initViewer()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install gepetto-viewer"
    )
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print(
        "Error while loading the viewer human_model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

lstm_dict = {**result_keypoints[0], **result_markers[0]}
jcp_planar = get_jcp_global_pos(lstm_dict, pos_ankle_calib)

for frame in human_model.frames.tolist():
    viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)

for jcp in jcp_planar.keys():
    viz.viewer.gui.addSphere('world/jcp_'+jcp, 0.01, [1,0.5,0,1])

# Not necessary, display markers + keypoints
# for keypoint in result_keypoints[0].keys():
#     viz.viewer.gui.addSphere('world/'+keypoint,0.01,[0,1,0,1])

# for marker in result_markers[0].keys():
#     viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])

for jj in range(q_to_plot.shape[0]):

    lstm_dict = {**result_keypoints[jj], **result_markers[jj]}
    jcp_planar = get_jcp_global_pos(lstm_dict, pos_ankle_calib)

    for key in jcp_planar.keys():
        M = pin.SE3(np.eye(3), np.array([jcp_planar[key][0],jcp_planar[key][1],jcp_planar[key][2]]))
        place(viz,'world/jcp_'+key,M)
    
    # for marker in result_markers[jj].keys():
    #     M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[jj][marker][0],result_markers[jj][marker][1],result_markers[jj][marker][2]]).T))
    #     place(viz,'world/'+marker,M)
    #     # input("Press Enter to continue...")

    # for marker in result_keypoints[jj].keys():
    #     M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_keypoints[jj][marker][0],result_keypoints[jj][marker][1],result_keypoints[jj][marker][2]]).T))
    #     place(viz,'world/'+marker,M)

    q0 =pin.neutral(human_model)
    q0[:]= q_to_plot[jj,:]

    viz.display(q0)

    pin.framesForwardKinematics(human_model,human_data, q0)

    for frame in human_model.frames.tolist():
        place(viz,'world/'+frame.name,human_data.oMf[human_model.getFrameId(frame.name)])
    
    input("Press Enter to continue...")
