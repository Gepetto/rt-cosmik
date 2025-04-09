#load mks data, get the barycenter. Load qrcode pose apply soder transformation and display everything in the same frame (mocap frame).
import sys
import os
import pinocchio as pin
import numpy as np
import pandas as pd
import time
from pinocchio.visualize import GepettoVisualizer
from utils import *

no_test = "calib_mocap_2_cam"
# mks_data_file = "/root/workspace/ros_ws/src/rt-cosmik/output/mks_data.csv"

mks_data_file =f"output/{no_test}/mks_data.csv"         # Contains markers data in one column ("mks_data")
aruco_data_file = f"output/{no_test}/pose_aruco.csv"    # Contains the ArUco point coordinates
transformation_file = f"output/{no_test}/soder.txt"  # Contains the transformation matrix and parameters

# Load transformation parameters (R, d, s, rms) from soder
R_trans, d_trans, s_trans, rms_error = load_transformation(transformation_file)
# Now, any point p in the other coordinate system is transformed via:
#   p_global = R_trans @ (p * s_trans) + d_trans

# Load marker data
df = pd.read_csv(mks_data_file)
# The mks_data column is assumed to be a string with semicolon-separated values.
mks_array = np.array([list(map(float, row.split(";"))) for row in df["mks_data"]])

# Load ArUco point data from CSV.
# Expected CSV header: timestamp,tvec_x,tvec_y,tvec_z,rvec_x,rvec_y,rvec_z
aruco_df = pd.read_csv(aruco_data_file)

# Initialize Gepetto Viewer
viz = GepettoVisualizer()

try:
    viz.initViewer()
    viz.loadViewerModel("pinocchio")
except Exception as err:
    print("Error while initializing or loading the viewer:", err)
    sys.exit(0)

# Add a base axis for reference
viz.viewer.gui.addXYZaxis('world/base_frame', [1, 0, 0, 1], 0.05, 0.3)


# Prepare spheres and frame in the viewer
viz.viewer.gui.addSphere('world/A', 0.05, [1, 0, 0, 1])         # Marker A: red
viz.viewer.gui.addSphere('world/B', 0.05, [0, 1, 0, 1])         # Marker B: green
viz.viewer.gui.addSphere('world/C', 0.05, [0, 0, 1, 1])         # Marker C: blue
viz.viewer.gui.addSphere('world/D', 0.05, [1, 1, 1, 1])         # Marker C: blue

viz.viewer.gui.addSphere('world/Barycenter', 0.01, [1, 1, 0, 1])  # Barycenter: yellow
viz.viewer.gui.addXYZaxis('world/local_frame', [1, 0, 1, 1], 0.03, 0.2)
viz.viewer.gui.addSphere('world/aruco', 0.01, [0.5, 0, 0.5, 1])   # ArUco point: purple

# A helper lambda to apply position updates
place = lambda name, pos: viz.viewer.gui.applyConfiguration(name, list(pos) + [0, 0, 0, 1])

# Assume the number of frames in mks_array and aruco_df match.
print(len(mks_array))
print(len(aruco_df))
num_frames = min(len(mks_array), len(aruco_df))

for i in range(num_frames):
    # Extract marker positions from mks_data
    A = mks_array[i, :3]
    B = mks_array[i, 3:6]
    C = mks_array[i, 6:9]
    D = mks_array[i, 9:]
    barycenter = (A + C) / 2
    R_local = calculate_frame(A, B, C)
    barycenter_local_frame = transform_to_local_frame(barycenter, B, R_local)
    barycenter_local_frame[2] = barycenter_local_frame[2]- 0.01
    barycenter_global_frame = transform_to_global_frame(barycenter_local_frame, B, R_local)

    # Display markers and barycenter
    place('world/A', A)
    place('world/B', B)
    place('world/C', C)
    place('world/D', D)

    place('world/Barycenter', barycenter_global_frame)
    # For the local frame, we update its origin at B.
    # Here we display an axis: we set its position to B and update orientation.
    # (For orientation, we convert the rotation matrix to a quaternion via pinocchio)
    quat = pin.Quaternion(R_local)
    viz.viewer.gui.applyConfiguration('world/local_frame', list(B) + list(quat.coeffs()))
    
    # Process the ArUco point data
    # Get the translation vector from the CSV (the point in the other coordinate system)
    tvec = np.array([aruco_df.loc[i, 'tvec_x'], aruco_df.loc[i, 'tvec_y'], aruco_df.loc[i, 'tvec_z']])
    # Transform it into the global frame
    # aruco_global = np.transpose(R_trans) @ (tvec - d_trans) if soder gives u mocap to cam
    aruco_global = R_trans @ tvec + d_trans  #if soder gives u cam to mocap
    place('world/aruco', aruco_global)
    
    viz.viewer.gui.refresh()
    time.sleep(0.03)  # Pause to visualize the update
    input("Press Enter to step to the next frame...")

