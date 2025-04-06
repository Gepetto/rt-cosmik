import os
import numpy as np
import torch
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.tools.vis_tools import colors, show_image
from body_visualizer.mesh.mesh_viewer import MeshViewer

# Choose device
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
amass_npz_fname = os.path.join(script_dir, 'amass', 'smplh', 'lift_box_poses.npz')
bdata = np.load(amass_npz_fname)
time_length = len(bdata['trans'])

subject_gender = bdata['gender'].item()
subject_gender = subject_gender.decode('utf-8') if isinstance(subject_gender, bytes) else subject_gender
print("Subject gender:", subject_gender)
print("Data keys:", list(bdata.keys()))
# print("num markers:", bdata['labels'])
num_betas = 16  # total shape coefficients
num_dmpls = 8   # number of DMPL parameters to use

# For SMPL-X, we only use the pose parameters.
# AMASS betas and DMPLs cannot be used with SMPL-X.
body_parms_smplx = {
    'root_orient': torch.tensor(bdata['poses'][:, :3], dtype=torch.float32).to(comp_device),
    'pose_body': torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(comp_device),
    'pose_hand': torch.tensor(bdata['poses'][:, 66:], dtype=torch.float32).to(comp_device),
    # 'trans': torch.tensor(bdata['trans'], dtype=torch.float32).to(comp_device),
    # # Repeat the betas for each frame since shape is static
    # 'betas': torch.tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis, :],
    #                                 repeats=time_length, axis=0), dtype=torch.float32).to(comp_device),
}
print({k: v.shape for k, v in body_parms_smplx.items()})

# Load the SMPL-X model.
bm_smplx_fname = os.path.join(script_dir, 'models', 'amass', 'smplx', subject_gender, 'model.npz')
bm = BodyModel(bm_fname=bm_smplx_fname, num_betas=16).to(comp_device)

faces = c2c(bm.f)
num_verts = bm.init_v_template.shape[1]


print("Number of vertices in the model:", num_verts)

# Forward pass: only pass pose parameters to SMPL-X
body = bm(**body_parms_smplx)

print("Mesh vertices shape for frame 0:", body.v[0].shape)
print("Mesh vertices shape for all frames:", body.v.shape)

# Visualize using body_visualizer
imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]),
                             faces=faces,
                             vertex_colors=np.tile(colors['grey'], (num_verts, 1)))
mv.set_static_meshes([body_mesh])
body_image = mv.render(render_wireframe=False)
show_image(body_image)
