import os
import numpy as np
import pickle as pk
import argparse
import torch
import pandas as pd


def read_pk_file(file_path, data_name='all'):
    """
    Lit et retourne les données d'un fichier .pk.

    :param file_path: Chemin vers le fichier .pk
    :param data_name: Type de données à lire
    :return: Données lues depuis le fichier
    """
    try:
        with open(file_path, 'rb') as file:
            data = pk.load(file)
            if data_name != 'all':
                data = data[data_name]
            return data
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'existe pas.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

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


# ----------------------------
# 1. Set up data path and script options
# ----------------------------

parser = argparse.ArgumentParser(description='Pickle/hybrik output reader and converter to csv markers, to vertices coord and to lstm markers addapted to pf')

parser.add_argument('--data-path',
                    help='data path',
                    dest='data_path',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    dest='out_dir',
                    help='output folder',
                    default='',
                    type=str)

opt = parser.parse_args()

cosmik_data_path = opt.data_path
out_dir = opt.out_dir

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
script_above_dir_path = script_dir.split('/')
marker_csv = '/' + os.path.join(*script_above_dir_path[:-1], 'amass', 'vertices_keypoints_corr.csv')
markers_df = pd.read_csv(marker_csv, delimiter=',')

# ----------------------------
# 2. Read marker indices from CSV and extract trajectories
# ----------------------------
# Assume the CSV file "markers.csv" is in the directory amass next to the script directory.
# The file is expected to have columns "Name" and "Index" separated by tabs.

counter_prog = 0
for subject in os.listdir(cosmik_data_path):
    subject_path = os.path.join(cosmik_data_path, subject)
    for trial in os.listdir(subject_path):
        trial_path = os.path.join(subject_path, trial)
        for element in os.listdir(trial_path):
            if ".pk" in element:
                pickle_path = os.path.join(trial_path, element)

                vertices = read_pk_file(pickle_path, data_name='pred_vertices') # shape: (num_frame, num_vertices, 3)
                print("Vertices array shape:", vertices.shape)

                # For each marker (vertex index), extract its 3D trajectory over time.

                # --- Marker trajectories extraction ---
                # 'vertices' is assumed to be a numpy array of shape (num_frames, num_vertices, 3)
                num_frames = vertices.shape[0]
                data_dict = {}
                # Read markers from the markers DataFrame (markers_df)
                for _, row in markers_df.iterrows():
                    marker_name = row['Name']
                    marker_index = int(row['Index'])
                    # Extract trajectory for this marker: shape (num_frames, 3)
                    traj = vertices[:, marker_index, :]
                    data_dict[f'{marker_name}_x'] = traj[:, 0]
                    data_dict[f'{marker_name}_y'] = traj[:, 1]
                    data_dict[f'{marker_name}_z'] = traj[:, 2]

                # Optionally, add a frame index column
                data_dict['Frame'] = np.arange(num_frames)

                # Create a DataFrame with columns: Frame, marker1_x, marker1_y, marker1_z, marker2_x, ...
                traj_df = pd.DataFrame(data_dict)
                # Reorder columns to have Frame first
                cols = ['Frame'] + [c for c in traj_df.columns if c != 'Frame']
                traj_df = traj_df[cols]

                # Save the marker trajectories to a CSV file.
                # motion_name is assumed to be defined elsewhere.
                output_csv = os.path.join(out_dir, subject, trial, f'{element[:-3]}_mks.csv')
                traj_df.to_csv(output_csv, index=False)
                print("Saved marker trajectories to", output_csv)

                # --- Vertices coordinates extraction ---
                # 'vertices' has shape (num_frames, num_vertices, 3)
                num_frames, num_vertices, _ = vertices.shape
                data_vertices = {}
                data_vertices['Frame'] = np.arange(num_frames)
                # For each vertex, add its x, y, and z coordinates across all frames.
                for v in range(num_vertices):
                    data_vertices[f'v_{v}_x'] = vertices[:, v, 0]
                    data_vertices[f'v_{v}_y'] = vertices[:, v, 1]
                    data_vertices[f'v_{v}_z'] = vertices[:, v, 2]

                vertices_df = pd.DataFrame(data_vertices)
                cols = ['Frame'] + [c for c in vertices_df.columns if c != 'Frame']
                vertices_df = vertices_df[cols]

                # Save the vertices trajectories to a CSV file.
                output_csv_vertices = os.path.join(out_dir, subject, trial, f'{element[:-3]}_vertices.csv')
                vertices_df.to_csv(output_csv_vertices, index=False)
                print("Saved vertices trajectories to", output_csv_vertices)


                # Path to CSV file with marker trajectories.
                input_csv = output_csv

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
                output_csv_lstm = os.path.join(out_dir, subject, trial, f'{element[:-3]}_mks_lstm.csv')
                df_formatted.to_csv(output_csv_lstm, index=False)

                print(f"✅ Formatted data saved to {output_csv_lstm}")

