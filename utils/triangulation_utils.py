import numpy as np
# from scipy import linalg
import cv2
# from scipy.spatial.transform import Rotation as R

def DLT_adaptive(projections, points):
    """
    Perform Direct Linear Transformation (DLT) for adaptive triangulation.
    This function computes the 3D coordinates of a point given its projections
    in multiple views using the DLT algorithm. It constructs a system of linear
    equations from the projection matrices and the corresponding 2D points, and
    then solves it using Singular Value Decomposition (SVD).
    Parameters:
    -----------
    projections : list of numpy.ndarray
        A list of 3x4 projection matrices for each view.
    points : list of numpy.ndarray
        A list of 2D points corresponding to each view. Each element in the list
        is an array of shape (n, 2), where n is the number of points.
    Returns:
    --------
    numpy.ndarray
        A 1D array of length 3 representing the 3D coordinates of the point.
    """
    
    A=[]
    for i in range(len(projections)):
        P=projections[i]
        point = points[i]

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    B = A.transpose() @ A
    _, _, Vh = np.linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def triangulate_points(keypoints_list, mtxs, dists, projections):
    """
    Triangulates 3D points from multiple 2D keypoints using camera matrices and distortion coefficients.
    Args:
        keypoints_list (list of list of tuples): A list where each element is a list of 2D keypoints for a single frame.
        mtxs (list of numpy.ndarray): A list of camera matrices for each frame.
        dists (list of numpy.ndarray): A list of distortion coefficients for each frame.
        projections (list of numpy.ndarray): A list of projection matrices for each frame.
    Returns:
        numpy.ndarray: An array of 3D points triangulated from the input 2D keypoints.
    """

    p3ds_frame=[]
    undistorted_points = []

    for ii in range(len(keypoints_list)):
        points = keypoints_list[ii] 
        distCoeffs_mat = np.array([dists[ii]]).reshape(-1, 1)
        points_undistorted = cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[ii], distCoeffs_mat)
        undistorted_points.append(points_undistorted)

    for point_idx in range(26):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        _p3d = DLT_adaptive(projections, points_per_point)
        p3ds_frame.append(_p3d)

    return np.array(p3ds_frame)
