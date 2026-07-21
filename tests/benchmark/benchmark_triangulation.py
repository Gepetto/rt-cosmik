"""Quick benchmark for triangulation backends.

Run:
    python tests/benchmark/benchmark_triangulation.py
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rtcosmik.triangulation.triangulation import DLT, triangulate_points, triangulate_points_torch


def make_data(num_cams: int = 4, num_joints: int = 26):
    f = 900.0
    cx = 640.0
    cy = 360.0
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    mtxs, dists, projections = [], [], []
    for i in range(num_cams):
        r = np.eye(3, dtype=np.float64)
        t = np.array([[i * 0.25], [0.02 * i], [0.0]], dtype=np.float64)
        projections.append(k @ np.hstack([r, t]))
        mtxs.append(k.copy())
        dists.append(np.zeros(5, dtype=np.float64))

    points_3d = np.random.uniform([-0.4, -0.4, 2.5], [0.4, 0.4, 5.5], size=(num_joints, 3))
    homog = np.hstack([points_3d, np.ones((num_joints, 1), dtype=np.float64)])

    keypoints_list = []
    for p in projections:
        uvw = (p @ homog.T).T
        keypoints_list.append(uvw[:, :2] / uvw[:, 2:3])

    return keypoints_list, mtxs, dists, projections


def triangulate_reference_loop(keypoints_list, mtxs, dists, projections):
    undistorted = []
    for i, points in enumerate(keypoints_list):
        dist_coeffs = np.array([dists[i]]).reshape(-1, 1)
        undistorted.append(cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[i], dist_coeffs))

    num_points = min(up.shape[0] for up in undistorted)
    out = []
    for j in range(num_points):
        out.append(DLT(projections, [undistorted[i][j] for i in range(len(undistorted))]))
    return np.array(out)


def main():
    for num_joints in [26, 133, 400]:
        keypoints_list, mtxs, dists, projections = make_data(num_cams=4, num_joints=num_joints)

        triangulate_reference_loop(keypoints_list, mtxs, dists, projections)
        triangulate_points(keypoints_list, mtxs, dists, projections)

        n_iter = 200 if num_joints <= 133 else 80

        t0 = time.perf_counter()
        for _ in range(n_iter):
            triangulate_reference_loop(keypoints_list, mtxs, dists, projections)
        t1 = time.perf_counter()

        for _ in range(n_iter):
            triangulate_points(keypoints_list, mtxs, dists, projections)
        t2 = time.perf_counter()

        loop_ms = ((t1 - t0) / n_iter) * 1e3
        numpy_ms = ((t2 - t1) / n_iter) * 1e3
        print(f"J={num_joints}: loop={loop_ms:.3f} ms, batched_numpy={numpy_ms:.3f} ms, speedup={loop_ms / numpy_ms:.2f}x")

    try:
        import torch

        if torch.cuda.is_available():
            keypoints_list, mtxs, dists, projections = make_data(num_cams=4, num_joints=133)
            und = []
            for i, points in enumerate(keypoints_list):
                dist_coeffs = np.array([dists[i]]).reshape(-1, 1)
                und.append(cv2.undistortPoints(np.array(points).reshape(-1, 1, 2), mtxs[i], dist_coeffs)[:, 0, :])

            points_t = torch.as_tensor(np.stack(und, axis=0), dtype=torch.float64, device="cuda")
            projections_t = torch.as_tensor(np.asarray(projections), dtype=torch.float64, device="cuda")

            # warmup
            triangulate_points_torch(points_t, projections_t, return_numpy=False)
            torch.cuda.synchronize()

            n_iter = 400
            t0 = time.perf_counter()
            for _ in range(n_iter):
                triangulate_points_torch(points_t, projections_t, return_numpy=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"Torch native on CUDA (J=133): {((t1 - t0) / n_iter) * 1e3:.3f} ms")
        else:
            print("Torch installed but CUDA not available; skipping CUDA benchmark.")
    except Exception as exc:
        print(f"Torch/CUDA benchmark skipped: {exc}")


if __name__ == "__main__":
    main()
