#load mks data, get the barycenter. Load qrcode pose apply soder transformation and display everything in the same frame (mocap frame).
import sys
import os
import pinocchio as pin
import numpy as np
import pandas as pd
import time
from pinocchio.visualize import GepettoVisualizer

mks_data_file = "output/test1/mks_data.csv"         # Contains markers data in one column ("mks_data")
# aruco_data_file = "/root/workspace/ros_ws/src/linear-algebra-toolkit/data/pose_aruco.csv"   
aruco_data_file = "output/test1/pose_aruco.csv"    # Contains the ArUco point coordinates
transformation_file = "output/test1/soder.txt"  # Contains the transformation matrix and parameters

# --- Load your transformation parameters using your own function ---
# (Assuming the load_transformation() function is defined somewhere accessible.)
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


# Load transformation parameters (R, d, s, rms)
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

# Function to compute local frame from three points
def calculate_frame(A, B, C):
    BA = A - B
    BC = C - B
    x = BA / np.linalg.norm(BA)
    y = BC / np.linalg.norm(BC)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

# Prepare spheres and frame in the viewer
viz.viewer.gui.addSphere('world/A', 0.01, [1, 0, 0, 1])         # Marker A: red
viz.viewer.gui.addSphere('world/B', 0.01, [0, 1, 0, 1])         # Marker B: green
viz.viewer.gui.addSphere('world/C', 0.01, [0, 0, 1, 1])         # Marker C: blue
viz.viewer.gui.addSphere('world/Barycenter', 0.01, [1, 1, 0, 1])  # Barycenter: yellow
viz.viewer.gui.addXYZaxis('world/local_frame', [1, 0, 1, 1], 0.03, 0.2)
viz.viewer.gui.addSphere('world/aruco', 0.01, [0.5, 0, 0.5, 1])   # ArUco point: purple

# A helper lambda to apply position updates
place = lambda name, pos: viz.viewer.gui.applyConfiguration(name, list(pos) + [0, 0, 0, 1])

# Assume the number of frames in mks_array and aruco_df match.
num_frames = min(len(mks_array), len(aruco_df))

for i in range(num_frames):
    # Extract marker positions from mks_data
    A = mks_array[i, :3]
    B = mks_array[i, 3:6]
    C = mks_array[i, -3:]
    barycenter = (A + C) / 2

    R_local = calculate_frame(A, B, C)

    # Display markers and barycenter
    place('world/A', A)
    place('world/B', B)
    place('world/C', C)
    place('world/Barycenter', barycenter)
    # For the local frame, we update its origin at B.
    # Here we display an axis: we set its position to B and update orientation.
    # (For orientation, we convert the rotation matrix to a quaternion via pinocchio)
    quat = pin.Quaternion(R_local)
    viz.viewer.gui.applyConfiguration('world/local_frame', list(B) + list(quat.coeffs()))
    
    # Process the ArUco point data
    # Get the translation vector from the CSV (the point in the other coordinate system)
    tvec = np.array([aruco_df.loc[i, 'tvec_x'], aruco_df.loc[i, 'tvec_y'], aruco_df.loc[i, 'tvec_z']])
    # Transform it into the global frame
    aruco_global = np.transpose(R_trans) @ (tvec - d_trans) 
    place('world/aruco', aruco_global)
    
    viz.viewer.gui.refresh()
    time.sleep(0.03)  # Pause to visualize the update
    input("Press Enter to step to the next frame...")

