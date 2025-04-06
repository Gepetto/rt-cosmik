import os
import numpy as np
import torch
import trimesh
import pandas as pd
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.tools.vis_tools import colors, show_image
from body_visualizer.mesh.mesh_viewer import MeshViewer

motion_name = 'lift_box_poses'

# ----------------------------
# 1. Set up data and model paths
# ----------------------------
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# Use your AMASS npz file (for SMPL-X, only pose parameters are used)
amass_npz_fname = os.path.join(script_dir, 'amass', 'smplh', f'{motion_name}.npz')
bdata = np.load(amass_npz_fname)
time_length = len(bdata['trans'])

# Get subject gender as a string
subject_gender = bdata['gender'].item()
subject_gender = subject_gender.decode('utf-8') if isinstance(subject_gender, bytes) else subject_gender
print("Subject gender:", subject_gender)
print("Data keys:", list(bdata.keys()))

num_betas = 16
# ----------------------------
# 2. Prepare pose parameters for SMPL-X
# (For SMPL-X we ignore betas and DMPLs from AMASS.)
# ----------------------------
body_parms_smplx = {
    'root_orient': torch.tensor(bdata['poses'][:, :3], dtype=torch.float32).to(comp_device),
    'pose_body': torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(comp_device),
    'pose_hand': torch.tensor(bdata['poses'][:, 66:], dtype=torch.float32).to(comp_device),
    'trans': torch.tensor(bdata['trans'], dtype=torch.float32).to(comp_device),
    # Repeat the betas for each frame since shape is static
    # 'betas': torch.tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis, :],
    #                                 repeats=time_length, axis=0), dtype=torch.float32).to(comp_device)
}
print({k: v.shape for k, v in body_parms_smplx.items()})

# ----------------------------
# 3. Load the SMPL-X model
# ----------------------------
bm_smplx_fname = os.path.join(script_dir, 'models', 'amass', 'smplx', subject_gender, 'model.npz')
bm = BodyModel(bm_fname=bm_smplx_fname, num_betas=16).to(comp_device)
faces = c2c(bm.f)
num_verts = bm.init_v_template.shape[1]
print("Number of vertices in the model:", num_verts)

# Forward pass: generate the mesh vertices using only the pose parameters.
body = bm(**body_parms_smplx)
print("Mesh vertices shape for frame 0:", body.v[0].shape)
print("Mesh vertices shape for all frames:", body.v.shape)

# ----------------------------
# 4. Visualize the mesh for the first frame (optional)
# ----------------------------
# imw, imh = 1600, 1600
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
# body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]),
#                             faces=faces,
#                             vertex_colors=np.tile(colors['grey'], (num_verts, 1)))
# mv.set_static_meshes([body_mesh])
# body_image = mv.render(render_wireframe=False)
# show_image(body_image)

# ----------------------------
# 5. Read marker indices from CSV and extract trajectories
# ----------------------------
# Assume the CSV file "markers.csv" is in the same directory as the script.
# The file is expected to have columns "Name" and "Index" separated by tabs.
marker_csv = os.path.join(script_dir, 'amass', 'vertices_keypoints_corr.csv')
markers_df = pd.read_csv(marker_csv, delimiter=',')

# Convert the model vertices to a numpy array.
# body.v is a tensor of shape (time_length, num_vertices, 3).
vertices = c2c(body.v)  # shape: (time_length, num_vertices, 3)
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
output_csv = os.path.join(script_dir, 'amass', 'smplh', f'{motion_name}_mks_2.csv')
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
output_csv_vertices = os.path.join(script_dir, 'amass', 'smplh', f'{motion_name}_vertices.csv')
vertices_df.to_csv(output_csv_vertices, index=False)
print("Saved vertices trajectories to", output_csv_vertices)

