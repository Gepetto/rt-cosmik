import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the npz file
file_path = os.path.join(script_dir, 'amass', 'smplh', 'lift_box_poses.npz')

# Load the .npz file
data = np.load(file_path)

print("Available keys and metadata in the .npz file:\n")
for key in data.keys():
    value = data[key]
    print(f"Key: '{key}'")
    print(f"  Type: {type(value)}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Shape: {value.shape}")
    print(f"  First element(s): {value[:1] if value.ndim > 0 else value}")
    print()
