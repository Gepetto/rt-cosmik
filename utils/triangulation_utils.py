import numpy as np
from scipy import linalg
import cv2
from scipy.spatial.transform import Rotation as R

def DLT_adaptive(projections, points):
    A=[]
    for i in range(len(projections)):
        P=projections[i]
        point = points[i]

        for j in range (len(point)):
            A.append(point[j][1]*P[2,:] - P[1,:])
            A.append(P[0,:] - point[j][0]*P[2,:])

    A = np.array(A).reshape((-1,4))
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def triangulate_points(keypoints_list, mtxs, dists, projections):
    p3ds_frame=[]
    undistorted_points = []

    for ii in range(len(keypoints_list)):
        points = keypoints_list[ii] 
        distCoeffs_mat = np.array([dists[ii]]).reshape(-1, 1)
        points_undistorted = cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[ii], distCoeffs_mat)
        undistorted_points.append(points_undistorted)

    for point_idx in range(17):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        _p3d = DLT_adaptive(projections, points_per_point)
        p3ds_frame.append(_p3d)

    return np.array(p3ds_frame)
