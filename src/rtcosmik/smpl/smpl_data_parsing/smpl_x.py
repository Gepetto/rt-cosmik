import os
import numpy as np
import torch
from smplx import SMPLX, SMPL, SMPLH

# Build the path to the npz file (adjust as needed)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'amass', 'smplh', 'lift_box_poses.npz')
data = np.load(file_path)

# Extract parameters for a single frame (e.g., frame 0)
pose = data['poses'][0]        # shape: (156,)
betas = data['betas'] #[:10]          # shape: (16,)
trans = data['trans'][0]       # shape: (3,)
gender = data['gender'].item() # e.g., 'female'

# Convert to torch tensors and add batch dimension
pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
trans_tensor = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)

# Specify the path to your SMPL-X model files
model_path = os.path.join(script_dir, 'models', 'smpl')  # update this path accordingly

# Initialize the SMPL-X model
model = SMPL(model_path, gender=gender, use_pca=False)

# Note:
# The SMPL-X model expects the pose parameters to be split into:
# - global_orient: first 3 values,
# - body_pose, jaw_pose, left_hand_pose, right_hand_pose, etc.
# The exact split depends on the SMPL-X configuration.
# For illustration, assuming the first 3 are global orientation:
global_orient = pose_tensor[:, :3]
body_pose = pose_tensor[:, 3:]  # the rest of the pose

# Forward pass to compute vertices
output = model(global_orient=global_orient, body_pose=body_pose,
               betas=betas_tensor, transl=trans_tensor)
vertices = output.vertices.detach().cpu().numpy()

print("Mesh vertices shape:", vertices.shape)
