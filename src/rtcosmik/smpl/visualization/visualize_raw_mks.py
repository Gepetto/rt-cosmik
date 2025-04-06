import os
import numpy as np
import pandas as pd
import time
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

# -----------------------------------------------------------------------------
# Function to load marker data from CSV
# -----------------------------------------------------------------------------
def load_marker_data(csv_file, marker_names):
    """
    Load marker data from a CSV file and return a list of dictionaries.
    
    The CSV is expected to have a header row where each marker's coordinates 
    are stored in columns with names like '<marker>_x', '<marker>_y', and '<marker>_z'.
    Any markers not found in the file are skipped.
    
    Parameters:
      csv_file (str): Path to the CSV file.
      marker_names (list of str): List of marker names to extract.
      
    Returns:
      markers_list (list of dict): Each element is a dict mapping marker name to
                                   a numpy array (3,) of its 3D position for that frame.
    """
    # Read the CSV file.
    data = pd.read_csv(csv_file)
    
    # Optionally, if the first column is a time/index column, drop it.
    # (Here we check if the first column's name contains "time" (case-insensitive)).
    if "time" in data.columns[0].lower():
        data = data.iloc[:, 1:]
    
    # Build a dictionary mapping marker name -> list of its coordinate column names.
    marker_columns = {}
    for marker in marker_names:
        # Look for columns that start with the marker name followed by an underscore.
        cols = [col for col in data.columns if col.lower().startswith(marker.lower() + "_")]
        if len(cols) < 3:
            print(f"Warning: Could not find 3 coordinate columns for marker '{marker}'.")
        else:
            # Sort columns so that _x, _y, _z are in order.
            # (This assumes the naming convention makes alphabetical order give x, then y, then z.)
            marker_columns[marker] = sorted(cols)[:3]
    
    num_frames = data.shape[0]
    markers_list = []
    
    # Loop through each frame (row in the CSV)
    for i in range(num_frames):
        frame_dict = {}
        for marker, cols in marker_columns.items():
            # Extract x, y, z values from the corresponding columns.
            try:
                x = data.iloc[i][cols[0]]
                y = data.iloc[i][cols[1]]
                z = data.iloc[i][cols[2]]
                frame_dict[marker] = np.array([x, y, z])
            except Exception as e:
                print(f"Error extracting marker '{marker}' in frame {i}: {e}")
        markers_list.append(frame_dict)
    
    return markers_list

# -----------------------------------------------------------------------------
# Function to animate markers in Gepetto Viewer
# -----------------------------------------------------------------------------
def animate_markers(viz, markers_list, marker_names, delay=0.03):
    """
    Animate markers in the Gepetto viewer.
    
    Parameters:
      viz: An initialized GepettoVisualizer instance.
      markers_list (list of dict): List of frames, each is a dict with marker positions.
      marker_names (list of str): List of marker names.
      delay (float): Time in seconds to wait between frames.
    """
    # Loop over each frame
    for frame in markers_list:
        for marker in marker_names:
            pos = frame[marker]
            # Create an SE3 transformation with identity rotation and the marker position as translation.
            T = pin.SE3(np.eye(3), pos.reshape(3, 1))
            sphere_name = 'world/' + marker
            # Update the marker sphere position using your helper "place" function.
            place(viz, sphere_name, T)
        time.sleep(delay)

# -----------------------------------------------------------------------------
# Example main routine
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys


    motion = 'lift_box_poses'
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(script_dir)
    
    # # SMPL marker names
    # marker_names = ['rshoulder', 'lshoulder', 'r_lelbow', 'l_lelbow',
    #                 'r_melbow', 'l_melbow', 'r_lwrist', 'l_lwrist', 'r_mwrist',
    #                 'l_mwrist', 'r_ASIS', 'l_ASIS', 'r_PSIS', 'l_PSIS', 'r_knee',
    #                 'l_knee', 'r_mknee', 'l_mknee', 'r_ankle', 'l_ankle', 'r_mankle',
    #                 'l_mankle', 'r_5meta', 'l_5meta', 'r_big_toe', 'l_big_toe', 'l_calc', 'r_calc', 'C7']
    
    
    #LSTM marker names
    # marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
    #                 'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
    #                 'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    #                 'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
    #                 'C7_study','r_lelbow_study', 'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
    #                 'L_lwrist_study','L_mwrist_study']

    #Full marker names
    marker_names = ['sternum', 'rshoulder', 'lshoulder', 'r_lelbow', 'l_lelbow', 'r_melbow', 'l_melbow', 'r_lwrist', 'l_lwrist', 
                    'r_mwrist', 'l_mwrist', 'r_ASIS', 'l_ASIS', 'r_PSIS', 'l_PSIS', 'r_knee', 'l_knee', 
                    'r_mknee', 'l_mknee', 'r_ankle', 'l_ankle', 'r_mankle', 'l_mankle', 'r_5meta', 
                    'l_5meta', 'r_toe', 'l_toe', 'r_big_toe', 'l_big_toe', 'l_calc', 'r_calc', 'r_bpinky', 
                    'l_bpinky', 'r_tpinky', 'l_tpinky', 'r_bindex', 'l_bindex', 'r_tindex', 'l_tindex', 
                    'r_tmiddle', 'l_tmiddle', 'r_tring', 'l_tring', 'r_bthumb', 'l_bthumb', 'r_tthumb', 
                    'l_tthumb', 'C7', 'L2', 'T11', 'T6'] 
    
    # Path to CSV file with marker trajectories.
    markers_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks_2.csv')
    markers_list = load_marker_data(markers_csv, marker_names)
    
    # Initialize the Gepetto Viewer
    viz = GepettoVisualizer()
    try:
        viz.initViewer()
    except ImportError as err:
        print("Error while initializing the viewer. Install gepetto-viewer.")
        sys.exit(0)
    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print("Error while loading the viewer model. Start gepetto-viewer.")
        sys.exit(0)
    
    # Add spheres for each marker with a given radius and color.
    # Here, as an example, all markers are added as blue spheres.
    for marker in marker_names:
        viz.viewer.gui.addSphere('world/' + marker, 0.01, [0, 0, 1, 1])
    
    # Optionally, add some coordinate axes
    viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0, 0, 1], 0.04, 0.2)
    
    # Animate the markers in the viewer
    animate_markers(viz, markers_list, marker_names, delay=0.03)
