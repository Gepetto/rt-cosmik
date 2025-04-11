import numpy as np
import pandas as pd
import cv2
import cv2 as cv
# Function to compute local frame from three points
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

# --- Load your transformation parameters using your own function ---
# (Assuming the load_transformation() function is defined somewhere accessible.)
def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms

def save_cam_to_cam_params(mtx1, dist1, mtx2, dist2, R, T, rmse, path):
    """
    Save stereo camera calibration parameters to a file.
    Args:
        mtx1 (numpy.ndarray): Camera matrix for the first camera.
        dist1 (numpy.ndarray): Distortion coefficients for the first camera.
        mtx2 (numpy.ndarray): Camera matrix for the second camera.
        dist2 (numpy.ndarray): Distortion coefficients for the second camera.
        R (numpy.ndarray): Rotation matrix between the two cameras.
        T (numpy.ndarray): Translation vector between the two cameras.
        rmse (float): Root Mean Square Error of the calibration.
        path (str): Path to the file where the parameters will be saved.
    Returns:
        None
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K1', mtx1)
    cv_file.write('D1', dist1)
    cv_file.write('K2', mtx2)
    cv_file.write('D2', dist2)
    cv_file.write('R', R)
    cv_file.write('T', T)
    cv_file.write('rmse', rmse)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def rotation_matrix_to_rodrigues(R):
    """Convert a rotation matrix to a Rodrigues rotation vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def rodrigues_to_matrix(rvec):
    """Convert Rodrigues rotation vector to rotation matrix."""
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    return R

def compute_relative_rotation(R1, R2):
    """Compute the relative rotation matrix between two rotation matrices."""
    return np.dot(R2, R1.T)