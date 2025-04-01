#get relative rot using qr code pose, relative rot using mks data to see if both data do the same rotation.
import numpy as np
import csv
from utils import *

no_test = 4
aruco_pose = f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/pose_aruco.csv" 
rot_aruco = f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/pose_aruco_rotation.csv"  
def process_csv(aruco_pose, rot_aruco):
    """Read a CSV file, convert Rodrigues rotation vectors to matrices, and save only relative rotations."""
    with open(aruco_pose, mode='r', newline='') as infile, open(rot_aruco, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header and write new header
        header = next(reader)
        writer.writerow(["vec1","vec2","vec3"])
        
        prev_R_matrix = None
        for row in reader:
            rvec = [float(row[4]), float(row[5]), float(row[6])]
            R_matrix = rodrigues_to_matrix(rvec)
            
            # Compute relative rotation if there is a previous rotation matrix
            if prev_R_matrix is not None:
                Rel_R = compute_relative_rotation(prev_R_matrix, R_matrix)
                Rel_rvec = rotation_matrix_to_rodrigues(Rel_R)

                writer.writerow(Rel_rvec.flatten().tolist())
            prev_R_matrix = R_matrix

process_csv(aruco_pose, rot_aruco)


rot_mocap = f"output/test{no_test}/mks_data_rotation.csv"
df = pd.read_csv(f"output/test{no_test}/mks_data.csv")
mks_array = np.array([list(map(float, row.split(";"))) for row in df["mks_data"]])

prev_R = None
rel_rotations = []  # List to store relative rotations
for i in range(len(mks_array)):
    C = mks_array[i, :3]
    B = mks_array[i, 3:6]
    A = mks_array[i, 6:9]
    
    # Calculate rotation matrix for the current frame
    R = calculate_frame(C, B, A)
    
    if prev_R is not None:
        # Compute relative rotation if there is a previous rotation matrix
        Rel_R = compute_relative_rotation(prev_R, R)
        Rel_rvec = rotation_matrix_to_rodrigues(Rel_R)
        rel_rotations.append(Rel_rvec.flatten().tolist())  # Store the relative rotation

    # Update the previous rotation matrix
    prev_R = R

# Save the relative rotations to a CSV
with open(rot_mocap, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["vec1","vec2","vec3"])  # Write header
    for rel_R in rel_rotations:
        writer.writerow(rel_R)

print(f"Relative rotations saved to {rot_mocap}")