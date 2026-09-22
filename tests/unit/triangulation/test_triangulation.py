import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from rtcosmik.triangulation.triangulation import DLT, triangulate_points, triangulate_points_torch


def _make_camera_matrices(num_cams: int = 3):
    mtxs = []
    dists = []
    projections = []

    f = 900.0
    cx = 640.0
    cy = 360.0
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    for i in range(num_cams):
        r = np.eye(3, dtype=np.float64)
        t = np.array([[i * 0.25], [0.0], [0.0]], dtype=np.float64)
        p = k @ np.hstack([r, t])
        mtxs.append(k.copy())
        dists.append(np.zeros(5, dtype=np.float64))
        projections.append(p)

    return mtxs, dists, projections


def _project_points(points_3d: np.ndarray, projections):
    keypoints_per_camera = []
    for p in projections:
        homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)])
        uvw = (p @ homog.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        keypoints_per_camera.append(uv)
    return keypoints_per_camera


def _triangulate_reference_loop(keypoints_list, mtxs, dists, projections):
    undistorted_points = []
    for cam_idx, points in enumerate(keypoints_list):
        dist_coeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
        points_undistorted = cv2.undistortPoints(
            np.array(points).reshape(-1, 1, 2),
            mtxs[cam_idx],
            dist_coeffs_mat,
        )
        undistorted_points.append(points_undistorted)

    num_points = min(up.shape[0] for up in undistorted_points)
    p3ds = []
    for point_idx in range(num_points):
        points_per_point = [undistorted_points[i][point_idx] for i in range(len(undistorted_points))]
        p3ds.append(DLT(projections, points_per_point))
    return np.array(p3ds)


def test_triangulate_points_matches_reference_loop_numpy():
    mtxs, dists, projections = _make_camera_matrices(num_cams=3)
    points_3d = np.array(
        [
            [0.10, -0.05, 3.50],
            [0.25, 0.12, 4.20],
            [-0.30, 0.18, 5.00],
            [0.05, -0.25, 2.80],
            [-0.15, 0.05, 3.80],
        ],
        dtype=np.float64,
    )
    keypoints_list = _project_points(points_3d, projections)

    reference = _triangulate_reference_loop(keypoints_list, mtxs, dists, projections)
    result = triangulate_points(keypoints_list, mtxs, dists, projections)

    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)


def test_triangulate_points_empty_input():
    mtxs, dists, projections = _make_camera_matrices(num_cams=2)

    result = triangulate_points([], mtxs, dists, projections)

    assert result.shape == (0, 3)


def test_triangulate_points_torch_native_matches_numpy_when_available():
    torch = pytest.importorskip("torch")

    mtxs, dists, projections = _make_camera_matrices(num_cams=3)
    points_3d = np.array(
        [
            [0.12, -0.07, 3.20],
            [0.08, 0.15, 4.10],
            [-0.21, 0.03, 5.30],
        ],
        dtype=np.float64,
    )
    keypoints_list = _project_points(points_3d, projections)

    # Build normalized undistorted points (C, J, 2) for the pure-torch path.
    und = []
    for cam_idx, points in enumerate(keypoints_list):
        dist_coeffs_mat = np.array([dists[cam_idx]]).reshape(-1, 1)
        und.append(
            cv2.undistortPoints(
                np.array(points).reshape(-1, 1, 2),
                mtxs[cam_idx],
                dist_coeffs_mat,
            )[:, 0, :]
        )

    points_cj2 = np.stack(und, axis=0)
    points_t = torch.as_tensor(points_cj2, dtype=torch.float64)
    projections_t = torch.as_tensor(np.asarray(projections, dtype=np.float64), dtype=torch.float64)

    result_torch_native = triangulate_points_torch(points_t, projections_t, return_numpy=True)
    result_numpy = triangulate_points(keypoints_list, mtxs, dists, projections)

    np.testing.assert_allclose(result_torch_native, result_numpy, rtol=1e-10, atol=1e-10)