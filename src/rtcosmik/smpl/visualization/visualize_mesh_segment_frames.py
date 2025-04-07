import os
import time
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from human_model.model_utils import construct_segments_frames  # Imported segment frames function

# -----------------------------
# Helper function to update sphere/axis pose.
def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

# -----------------------------
# Function to load marker data from CSV
def load_marker_data(csv_file, marker_names):
    """
    Load marker data from a CSV file and return a list of dictionaries.
    
    The CSV is expected to have a header row where each marker's coordinates 
    are stored in columns with names like '<marker>_x', '<marker>_y', and '<marker>_z'.
    Any markers not found in the file are skipped.
    """
    data = pd.read_csv(csv_file)
    # If the first column is a time column, drop it.
    if "time" in data.columns[0].lower():
        data = data.iloc[:, 1:]
    
    # Build a dictionary mapping marker name -> list of coordinate columns.
    marker_columns = {}
    for marker in marker_names:
        cols = [col for col in data.columns if col.lower().startswith(marker.lower() + "_")]
        if len(cols) < 3:
            print(f"Warning: Could not find 3 coordinate columns for marker '{marker}'.")
        else:
            marker_columns[marker] = sorted(cols)[:3]
    
    num_frames = data.shape[0]
    markers_list = []
    for i in range(num_frames):
        frame_dict = {}
        for marker, cols in marker_columns.items():
            try:
                x = data.iloc[i][cols[0]]
                y = data.iloc[i][cols[1]]
                z = data.iloc[i][cols[2]]
                frame_dict[marker] = np.array([x, y, z])
            except Exception as e:
                print(f"Error extracting marker '{marker}' in frame {i}: {e}")
        markers_list.append(frame_dict)
    
    return markers_list

# -----------------------------
# Function to load vertices data from CSV.
def load_vertices_data(csv_file):
    """
    Load vertices data from a CSV file.
    
    The CSV is expected to have a "Frame" column, and then columns:
      v_0_x, v_0_y, v_0_z, v_1_x, v_1_y, v_1_z, ..., etc.
    
    Returns:
      vertices: A numpy array of shape (num_frames, num_vertices, 3)
    """
    df = pd.read_csv(csv_file)
    if 'Frame' in df.columns:
        df = df.drop(columns=['Frame'])
    num_cols = df.shape[1]
    num_vertices = num_cols // 3
    vertices = df.to_numpy().reshape(-1, num_vertices, 3)
    return vertices

# -----------------------------
# Function to animate markers, vertices, and segments in Gepetto Viewer.
def animate_all(viz, markers_list, vertices, marker_names, vertex_factor=1, delay=0.03):
    """
    Animate markers, vertices, and body segments in Gepetto Viewer.
    
    For each frame:
      - Marker spheres are updated using marker positions.
      - Vertices (displayed as small spheres) are updated.
      - Segment frames are computed from marker positions and coordinate axes are updated.
    """
    num_frames, num_vertices, _ = vertices.shape
    selected_indices = np.arange(0, num_vertices, vertex_factor)
    
    # Create the initial vertex spheres.
    for idx in selected_indices:
        viz.viewer.gui.addSphere(f'world/v_{idx}', 0.005, [0, 0, 1, 0.6])
    
    # Optionally add marker spheres if not already added.
    for marker in marker_names:
        viz.viewer.gui.addSphere('world/' + marker, 0.008, [1, 0, 0, 1])
    
    # Compute segments from the first frame and add coordinate axes for each segment.
    initial_segments = construct_segments_frames(markers_list[0], with_head=False)
    for seg_name in initial_segments.keys():
        viz.viewer.gui.addXYZaxis('world/' + seg_name, [255, 0, 0, 1], 0.01, 0.08)
    
    # Animation loop.
    for i in range(num_frames):
        # Update markers.
        for marker in marker_names:
            if marker in markers_list[i]:
                pos = markers_list[i][marker]
                T = pin.SE3(np.eye(3), pos.reshape(3, 1))
                place(viz, 'world/' + marker, T)
            else:
                print(f"Warning: Marker '{marker}' missing in frame {i}.")
        
        # Update vertices.
        for idx in selected_indices:
            pos = vertices[i, idx, :]
            T = pin.SE3(np.eye(3), pos.reshape(3, 1))
            place(viz, f'world/v_{idx}', T)
        
        # Compute segment frames from marker data and update their axes.
        segments = construct_segments_frames(markers_list[i], with_head=False)
        for seg_name, pose in segments.items():
            # Convert the 4x4 pose to a pin.SE3 object.
            M = pin.SE3(pose[:3, :3], pose[:3, 3].reshape(3, 1))
            place(viz, 'world/' + seg_name, M)
        
        time.sleep(delay)

# -----------------------------
# Main routine.
if __name__ == '__main__':
    import sys

    motion = 'lift_box_poses'
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(script_dir)
    
    # Define marker names of interest.
    # marker_names = ['rshoulder', 'lshoulder', 'r_lelbow', 'l_lelbow',
    #                 'r_melbow', 'l_melbow', 'r_lwrist', 'l_lwrist', 'r_mwrist',
    #                 'l_mwrist', 'r_ASIS', 'l_ASIS', 'r_PSIS', 'l_PSIS', 'r_knee',
    #                 'l_knee', 'r_mknee', 'l_mknee', 'r_ankle', 'l_ankle', 'r_mankle',
    #                 'l_mankle', 'r_5meta', 'l_5meta', 'r_big_toe', 'l_big_toe', 'l_calc', 'r_calc', 'C7']

    #LSTM marker names
    marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
                    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
                    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
                    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
                    'C7_study','r_lelbow_study', 'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
                    'L_lwrist_study','L_mwrist_study']
    
    # Path to CSV file with marker trajectories.
    markers_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks_lstm.csv')
    markers_list = load_marker_data(markers_csv, marker_names)


    # Path to CSV file with vertices trajectories.
    vertices_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_vertices.csv')
    vertices = load_vertices_data(vertices_csv)
    print("Loaded vertices with shape:", vertices.shape)
    
    # Initialize Gepetto Viewer.
    viz = GepettoVisualizer()
    try:
        viz.initViewer()
    except ImportError as err:
        print("Error initializing viewer. Install gepetto-viewer.")
        sys.exit(0)
    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print("Error loading viewer model. Start gepetto-viewer.")
        sys.exit(0)
    
    # Set viewer background colors.
    viz.viewer.gui.setBackgroundColor1('python-pinocchio', [1, 1, 1, 1])
    viz.viewer.gui.setBackgroundColor2('python-pinocchio', [1, 1, 1, 1])
    
    # Animate markers, vertices, and segments.
    animate_all(viz, markers_list, vertices, marker_names, vertex_factor=1, delay=0.03)
