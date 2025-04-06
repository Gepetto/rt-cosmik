import os
import time
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

# -----------------------------
# Helper function to update sphere pose.
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
        frames = df['Frame'].values  # (num_frames,)
        df = df.drop(columns=['Frame'])
    else:
        frames = None
    # Assume remaining columns are ordered so that every three columns form x,y,z.
    num_cols = df.shape[1]
    num_vertices = num_cols // 3
    vertices = df.to_numpy().reshape(-1, num_vertices, 3)
    return vertices

# -----------------------------
# Function to animate markers and vertices in Gepetto Viewer.
def animate_all(viz, markers_list, vertices, marker_names, vertex_factor=1, delay=0.001):
    """
    Animate markers and vertices in Gepetto Viewer.
    
    Instead of updating each vertex sphere individually, this version updates a
    single point cloud object for the vertices.
    """
    num_frames, num_vertices, _ = vertices.shape
    # Compute indices of vertices to display.
    selected_indices = np.arange(0, num_vertices, vertex_factor)
    
    # Create the initial point cloud object.
    # Gather the selected vertices from the first frame.
    selected_points = vertices[0, selected_indices, :]  # shape (n_sel, 3)
    
    # Create a single mesh (point cloud) from these points.
    # Here, we simply use spheres for each point.
    # Many viewers allow to display point clouds directly, but with Gepetto,
    # one approach is to create a mesh where each vertex is rendered as a small sphere.
    # For simplicity, we aggregate them into one mesh.
    # Create one sphere for each selected vertex? Alternatively, create a custom
    # shape representing a point cloud. For now, we assume a custom function exists.
    # Here, we add one sphere per selected vertex.
    for idx in selected_indices:
        viz.viewer.gui.addSphere(f'world/v_{idx}', 0.002, [0, 0, 1, 0.8])
    
    # Animation loop.
    for i in range(num_frames):
        # Update markers.
        for marker in marker_names:
            pos = markers_list[i][marker]
            T = pin.SE3(np.eye(3), pos.reshape(3,1))
            place(viz, 'world/' + marker, T)
        # Update the point cloud: update each selected vertex sphere.
        for idx in selected_indices:
            pos = vertices[i, idx, :]
            T = pin.SE3(np.eye(3), pos.reshape(3,1))
            place(viz, f'world/v_{idx}', T)
        time.sleep(delay)

# -----------------------------
# Main routine.
if __name__ == '__main__':
    import sys

    motion = 'lift_box_poses'
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(script_dir)
    
    # Define marker names of interest.
    marker_names = ['rshoulder', 'lshoulder', 'r_lelbow', 'l_lelbow',
                    'r_melbow', 'l_melbow', 'r_lwrist', 'l_lwrist', 'r_mwrist',
                    'l_mwrist', 'r_ASIS', 'l_ASIS', 'r_PSIS', 'l_PSIS', 'r_knee',
                    'l_knee', 'r_mknee', 'l_mknee', 'r_ankle', 'l_ankle', 'r_mankle',
                    'l_mankle', 'r_5meta', 'l_5meta', 'r_big_toe', 'l_big_toe', 'l_calc', 'r_calc', 'C7']
    
    # Path to CSV file with marker trajectories.
    markers_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks.csv')
    markers_list = load_marker_data(markers_csv, marker_names)
    
    # Path to CSV file with vertices trajectories.
    vertices_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_vertices.csv')
    vertices = load_vertices_data(vertices_csv)
    print("Loaded vertices with shape:", vertices.shape)
    
    # Initialize Gepetto Viewer.
    from pinocchio.visualize import GepettoVisualizer
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
    
    # Add marker spheres.
    for marker in marker_names:
        viz.viewer.gui.addSphere('world/' + marker, 0.008, [1, 0, 0, 1])
    
    # viz.viewer.gui()
    # Add small vertex spheres.
    num_vertices = vertices.shape[1]
    vertex_factor=1
    viz.viewer.gui.setBackgroundColor1('python-pinocchio', [1, 1, 1, 1])
    viz.viewer.gui.setBackgroundColor2('python-pinocchio', [1, 1, 1, 1])
    # selected_indices = np.arange(0, num_vertices, vertex_factor)
    # for j in selected_indices:
    #     viz.viewer.gui.addSphere(f'world/v_{j}', 0.002, [0, 0, 1, 0.8])
    
    # # Optionally add coordinate axes.
    # viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0, 0, 1], 0.04, 0.2)
    
    # Animate both markers and vertices.
    
    animate_all(viz, markers_list, vertices, marker_names, vertex_factor=vertex_factor, delay=0.03)
