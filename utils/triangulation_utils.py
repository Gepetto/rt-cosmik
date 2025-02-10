import numpy as np
from scipy import linalg
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




def triangulate_points_off(uvs, mtxs, dists, projections):
    num_frames = len(uvs[0])  # Nombre de frames, basé sur la première caméra
    print(num_frames)
    num_points = len(uvs[0][0])  # Nombre de points, basé sur la première frame de la première caméra
    p3ds_frames = []

    for frame_idx in range(num_frames):

        p3ds_frame=[]
        points_2d_per_frame = [uv[frame_idx] for uv in uvs] 

        undistorted_points = []

        for cam_idx in range(len(points_2d_per_frame)):
            points = points_2d_per_frame[cam_idx]
            distCoeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
            points_undistorted = cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[cam_idx], distCoeffs_mat)

            undistorted_points.append(points_undistorted)

        for point_idx in range(num_points):
            points_per_point = [undistorted_points[cam_idx][point_idx] for cam_idx in range(len(undistorted_points))]
            _p3d = DLT(projections, points_per_point)
            p3ds_frame.append(_p3d)

        p3ds_frames.append(p3ds_frame)

    return np.array(p3ds_frames)


def DLT(projections, points):
    num_camera=len(projections)
    A=[]
    for i in range (num_camera):
        P=projections[i]
        point = points[i]
        # print ('point',point)

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    #print('A: ')
    #print(A)
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]
