import numpy as np
import pandas as pd

def calculate_frame(A, B, C):
    # Step 1: Compute the vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Step 2: Define the x-axis (direction from B to A) and normalize it
    x = BA / np.linalg.norm(BA)
    
    # Step 3: Define the y-axis (direction from B to C) and normalize it
    y = BC / np.linalg.norm(BC)
    
    # Step 4: Compute the z-axis (cross product of x and y, then normalize it)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    y = np.cross(z,x)
    
    # Step 5: Construct the rotation matrix
    rotation_matrix = np.column_stack((x, y, z))
    
    return rotation_matrix

def transform_to_local_frame(D, origin, rotation_matrix):
    # Compute D relative to B
    D_relative = D - origin
    
    # Transform D to the local frame
    D_local = rotation_matrix.T @ D_relative
    
    return D_local

def transform_to_global_frame(D, origin, rotation_matrix):

    D_global =  rotation_matrix @ D + origin
    return D_global


df = pd.read_csv("output/test1/mks_data.csv")
mks_array = np.array([list(map(float, row.split(";"))) for row in df["mks_data"]])
barycenter_global_list = []
  
for i in range (len(mks_array)):
    A = mks_array[i, :3]
    B = mks_array[i, 3:6]
    C = mks_array[i, -3:]
    R = calculate_frame(A, B, C)
    # print(A)
    # print(C)

    barycenter = (A + C) / 2
    # barycenter_local_frame = transform_to_local_frame(barycenter, B, R)
    # barycenter_local_frame[2] = barycenter_local_frame[2]- 0.01
    # print(barycenter_local_frame)


    # barycenter_global_frame = transform_to_global_frame(barycenter_local_frame, B, R)
    # print(barycenter_global_frame)
    barycenter_global_list.append(barycenter)

# Convert to DataFrame and save to CSV
barycenter_global_df = pd.DataFrame(barycenter_global_list, columns=["Bx", "By", "Bz"])
barycenter_global_df.to_csv("output/test1//barycenter_raw.csv", index=False)

