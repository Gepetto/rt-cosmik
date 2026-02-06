import numpy as np
import cv2
from scipy import linalg
import torch

# from scipy.spatial.transform import Rotation as R

def DLT(projections, points):
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

    if len(undistorted_points) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Remove hard-coded number of keypoints.
    # Use the minimum number of points across cameras to avoid index errors.
    num_points = min(up.shape[0] for up in undistorted_points)

    for point_idx in range(num_points):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        _p3d = DLT(projections, points_per_point)
        p3ds_frame.append(_p3d)

    return np.array(p3ds_frame)

def triangulate_offline(uvs, mtxs, dists, projections, R, T):
    """Triangulate and transform keypoints for all frames."""
    keypoints_in_world_list = []
    num_frames = len(uvs[0])
    
    for frame_idx in range(num_frames):
        points_2d_per_frame = [uv[frame_idx] for uv in uvs]
        p3d_frame = triangulate_points(points_2d_per_frame, mtxs, dists, projections) #3d is expressed in cam0 frame
        #express p3d_frame in world frame
        p3d_frame_in_world= np.array([np.dot(R, point) + T for point in p3d_frame]) #if we have an external frame
        keypoints_in_world_list.append(p3d_frame.flatten().tolist())
    
    return keypoints_in_world_list

def DLT_adaptive(projections, points, idx_cams_used: list):
    A=[]
    for i in range(len(idx_cams_used)):
        P=projections[idx_cams_used[i]]
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

def is_camera_used(score: float, threshold: float)->bool:
    if score > threshold:
        return True
    else:
        return False

def which_cameras_used(scores: list, threshold: float)->list:
    which_cam_used_list = []
    for score in scores:
        which_cam_used_list.append(is_camera_used(score, threshold))
    return which_cam_used_list

def index_cameras_used(which_cam_used_list: list):
    index_cameras_used = []
    for i in range(len(which_cam_used_list)):
        if which_cam_used_list[i] == True:
            index_cameras_used.append(i)
    return index_cameras_used

def triangulate_points_adaptive(uvs, mtxs, dists, projections, scores: list, threshold: float):
    num_frames = len(uvs[0])  # Nombre de frames, basé sur la première caméra
    num_points = len(uvs[0][0])  # Nombre de points, basé sur la première frame de la première caméra
    p3ds_frames = []
    keypoints_in_cam0_list = []

    for frame_idx in range(num_frames):
        which_cam_used_list = which_cameras_used(scores[frame_idx], threshold)
        p3ds_frame=[]
        points_2d_per_frame = [uv[frame_idx] for uv in uvs] 

        undistorted_points = []

        idx_cams_used = index_cameras_used(which_cam_used_list)
        print(idx_cams_used)
        for cam_idx in idx_cams_used:
            points = points_2d_per_frame[cam_idx]
            distCoeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
            points_undistorted = cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[cam_idx], distCoeffs_mat)
            undistorted_points.append(points_undistorted)

        for point_idx in range(num_points):
            points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
            _p3d = DLT_adaptive(projections, points_per_point, idx_cams_used)
            p3ds_frame.append(_p3d)

        
        # On ajoute la version aplatie
        keypoints_in_cam0_list.append(np.array(p3ds_frame).flatten().tolist())

    return keypoints_in_cam0_list

def project_points_cam_to_pixels(points_cam: torch.Tensor, K: np.ndarray) -> np.ndarray:
    pts = points_cam.detach().float().cpu().numpy()   # (J,3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return np.stack([u, v], axis=1)                   # (J,2)

