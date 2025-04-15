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


# ----------------------------
# 1. Set up data path and script options
# ----------------------------

parser = argparse.ArgumentParser(description='Pickle/hybrik output reader and converter to csv markers')

parser.add_argument('--data-path',
                    help='data path',
                    dest='data_path',
                    default='',
                    type=str)
# parser.add_argument('--out-dir',
#                     dest='out_dir',
#                     help='output folder',
#                     default='',
#                     type=str)

opt = parser.parse_args()

data_path = opt.data_path
print(data_path)
# out_dir = opt.out_dir
name_exp = data_path.split('/')[-1][:-3]


# ----------------------------
# 2. Read marker indices from CSV and extract trajectories
# ----------------------------
# Assume the CSV file "markers.csv" is in the directory amass next to the script directory.
# The file is expected to have columns "Name" and "Index" separated by tabs.

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
script_above_dir_path = script_dir.split('/')

marker_csv = '/' + os.path.join(*script_above_dir_path[:-1], 'amass', 'vertices_keypoints_corr.csv')
markers_df = pd.read_csv(marker_csv, delimiter=',')

vertices = read_pk_file(data_path, data_name='pred_vertices') # shape: (num_frame, num_vertices, 3)
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
output_csv = "/" + os.path.join(*script_above_dir_path[:-2], 'smpl', 'hybrik', 'smplx', f'{name_exp}', f'{name_exp}_mks_2.csv')
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
output_csv_vertices = "/" + os.path.join(*script_above_dir_path[:-2], 'smpl', 'hybrik', 'smplx', f'{name_exp}', f'{name_exp}_vertices.csv')
vertices_df.to_csv(output_csv_vertices, index=False)
print("Saved vertices trajectories to", output_csv_vertices)

