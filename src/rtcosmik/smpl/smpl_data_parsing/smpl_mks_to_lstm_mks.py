import pandas as pd
import numpy as np

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
    if "time" in data.columns[0].lower():
        data = data.iloc[:, 1:]
    
    # Build a dictionary mapping marker name -> list of its coordinate column names.
    marker_columns = {}
    for marker in marker_names:
        cols = [col for col in data.columns if col.lower().startswith(marker.lower() + "_")]
        if len(cols) < 3:
            print(f"Warning: Could not find 3 coordinate columns for marker '{marker}'.")
        else:
            # Assumes alphabetical order gives x, then y, then z.
            marker_columns[marker] = sorted(cols)[:3]
    
    num_frames = data.shape[0]
    markers_list = []
    
    # Loop through each frame (row in the CSV)
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

# --- Define marker names and mappings ---

# Original marker names (in data order)
marker_names = ['rshoulder', 'lshoulder', 'r_lelbow', 'l_lelbow',
                'r_melbow', 'l_melbow', 'r_lwrist', 'l_lwrist', 'r_mwrist',
                'l_mwrist', 'r_ASIS', 'l_ASIS', 'r_PSIS', 'l_PSIS', 'r_knee',
                'l_knee', 'r_mknee', 'l_mknee', 'r_ankle', 'l_ankle', 'r_mankle',
                'l_mankle', 'r_5meta', 'l_5meta', 'r_big_toe', 'l_big_toe', 'l_calc', 'r_calc', 'C7']

# Desired final marker names (target order)
new_marker_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
                    'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
                    'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
                    'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
                    'C7_study','r_lelbow_study', 'r_melbow_study','r_lwrist_study','r_mwrist_study',
                    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study']

# Mapping from original marker names to new marker names
name_map = {
    'r_ASIS': 'r.ASIS_study',
    'l_ASIS': 'L.ASIS_study',
    'r_PSIS': 'r.PSIS_study',
    'l_PSIS': 'L.PSIS_study',
    'r_knee': 'r_knee_study',
    'r_mknee': 'r_mknee_study',
    'r_ankle': 'r_ankle_study',
    'r_mankle': 'r_mankle_study',
    'r_big_toe': 'r_toe_study',
    'r_5meta': 'r_5meta_study',
    'r_calc': 'r_calc_study',
    'l_knee': 'L_knee_study',
    'l_mknee': 'L_mknee_study',
    'l_ankle': 'L_ankle_study',
    'l_mankle': 'L_mankle_study',
    'l_big_toe': 'L_toe_study',
    'l_calc': 'L_calc_study',
    'l_5meta': 'L_5meta_study',
    'rshoulder': 'r_shoulder_study',
    'lshoulder': 'L_shoulder_study',
    'C7': 'C7_study',
    'r_lelbow': 'r_lelbow_study',
    'r_melbow': 'r_melbow_study',
    'r_lwrist': 'r_lwrist_study',
    'r_mwrist': 'r_mwrist_study',
    'l_lelbow': 'L_lelbow_study',
    'l_melbow': 'L_melbow_study',
    'l_lwrist': 'L_lwrist_study',
    'l_mwrist': 'L_mwrist_study'
}

# Build an inverse mapping: new marker name -> original marker name
inverse_name_map = {new: old for old, new in name_map.items()}

# --- Load data and reformat ---
import sys
import os

motion = 'lift_box_poses'
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(script_dir)

# Path to CSV file with marker trajectories.
input_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks_2.csv')

# Load only the desired markers from the CSV file.
markers_list = load_marker_data(input_csv, marker_names)

# Create a new list with the new marker names and order.
formatted_markers_list = []
for frame in markers_list:
    formatted_frame = {}
    for new_marker in new_marker_names:
        # Find the corresponding original marker name.
        old_marker = inverse_name_map.get(new_marker)
        if old_marker is None:
            print(f"Warning: No mapping found for new marker '{new_marker}'.")
            continue
        # Only add the marker if it exists in the current frame.
        if old_marker in frame:
            formatted_frame[new_marker] = frame[old_marker]
        else:
            print(f"Warning: Marker '{old_marker}' not found in frame data.")
    formatted_markers_list.append(formatted_frame)

# Prepare data for CSV output.
# Each frame will become a row and each marker's coordinates are split into _x, _y, _z columns.
rows = []
for frame in formatted_markers_list:
    row = {}
    for marker in new_marker_names:
        if marker in frame:
            coords = frame[marker]
            row[f"{marker}_x"] = coords[0]
            row[f"{marker}_y"] = coords[1]
            row[f"{marker}_z"] = coords[2]
        else:
            # If marker data is missing, fill with NaN.
            row[f"{marker}_x"] = np.nan
            row[f"{marker}_y"] = np.nan
            row[f"{marker}_z"] = np.nan
    rows.append(row)

df_formatted = pd.DataFrame(rows)

# Write the formatted data to a new CSV file.
output_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion}_mks_lstm.csv')
df_formatted.to_csv(output_csv, index=False)

print(f"✅ Formatted data saved to {output_csv}")
