import os
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Choose the device (use GPU if available)
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update this path to where you have your support data (model files, sample npz, etc.)
# For example, suppose you have a folder structure:
#   <project_root>/
#       support_data/
#           github_data/dmpl_sample.npz
#           body_models/smplh/male/model.npz
#           body_models/dmpls/male/model.npz
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Path to a sample AMASS npz file (adjust filename as needed)
amass_npz_fname = os.path.join(script_dir, 'amass', 'smplh', 'lift_box_poses.npz')
bdata = np.load(amass_npz_fname)

# Get the subject's gender from the npz file
subject_gender = bdata['gender'].item()
subject_gender = subject_gender.decode('utf-8') if isinstance(subject_gender, bytes) else subject_gender

print('Subject gender:', subject_gender)

print('Data keys:', list(bdata.keys()))
# print("num markers:", bdata['labels'])
# Define the number of shape and DMPL coefficients
num_betas = 16  # total shape coefficients
num_dmpls = 8   # number of DMPL parameters to use

# Paths to the body model (SMPL+H) and DMPL model files
bm_fname = os.path.join(script_dir, 'models', 'amass', 'smplh', subject_gender, 'model.npz')
dmpl_fname = os.path.join(script_dir, 'models', 'amass', 'dmpls', subject_gender, 'model.npz')

# Initialize the BodyModel with DMPLs
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls,
               dmpl_fname=dmpl_fname).to(comp_device)

# Get the mesh faces from the body model
faces = c2c(bm.f)

# Determine the number of frames in the sequence (time length)
time_length = len(bdata['trans'])
print('Time length (number of frames):', time_length)

# Create a dictionary of body parameters
# Here, the AMASS poses are split into:
#   - root_orient: first 3 values (global orientation)
#   - pose_body: next 63 values (21 joints * 3) controlling the body (without hands)
#   - pose_hand: remaining values controlling hand articulation
# Adjust these indices if your dataset uses a different convention.
body_parms = {
    'root_orient': torch.tensor(bdata['poses'][:, :3], dtype=torch.float32).to(comp_device),
    'pose_body': torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(comp_device),
    'pose_hand': torch.tensor(bdata['poses'][:, 66:], dtype=torch.float32).to(comp_device),
    'trans': torch.tensor(bdata['trans'], dtype=torch.float32).to(comp_device),
    # Repeat the betas for each frame since shape is static
    'betas': torch.tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis, :],
                                    repeats=time_length, axis=0), dtype=torch.float32).to(comp_device),
    # Use only the first num_dmpls DMPL coefficients
    'dmpls': torch.tensor(bdata['dmpls'][:, :num_dmpls], dtype=torch.float32).to(comp_device)
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))

# Forward pass: compute the mesh for all frames
body_model_output = bm(**body_parms)

# For example, get the mesh vertices for the first frame:
vertices_frame0 = c2c(body_model_output.v[0])
print("Mesh vertices shape for frame 0:", vertices_frame0.shape)
print("Mesh vertices shape for all frames:", body_model_output.v.shape)


# (Optional) Visualize the mesh using trimesh and body_visualizer if desired:
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas']})

def vis_body_pose_beta(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

vis_body_pose_beta(fId=0)

