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

def _two_views():
    from rtcosmik.nlf.nlf import Views
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    poses = [np.array([[0.0, 0.0, 2.0]]), np.array([[0.3, 0.0, 2.0]])]
    sigma = np.array([[1.0], [3.0]])
    return Views([None, None], poses, sigma, [0, 1]), [identity, identity]


def test_views_are_fused_by_their_mean_by_default():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    views, projections = _two_views()
    np.testing.assert_allclose(reconstruct_3d(views, projections), [[0.15, 0.0, 2.0]])


def test_inverse_variance_fusion_favours_the_certain_view():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    views, projections = _two_views()
    fused = reconstruct_3d(views, projections, weighting="invvar")
    np.testing.assert_allclose(fused, [[0.03, 0.0, 2.0]])


def test_unknown_fusion_rule_is_refused():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    views, projections = _two_views()
    with pytest.raises(ValueError):
        reconstruct_3d(views, projections, weighting="mode")


def test_settings_fuse_by_the_mean():
    from rtcosmik.config_loader import settings
    assert settings.fusion == "mean"


def _four_views(x_offsets, sigma=None):
    """One landmark at (0.2, -0.1, 2.5) m in the reference frame; view i reads it
    ``x_offsets[i]`` metres off along x. Views 1-3 are rotated and translated."""
    from rtcosmik.nlf.nlf import Views
    truth = np.array([0.2, -0.1, 2.5])
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = [np.eye(3), quarter, quarter @ quarter, quarter.T]
    translations = [np.zeros(3), np.array([0.5, 0.0, 0.1]), np.array([0.0, 1.0, 0.0]),
                    np.array([-0.3, 0.2, 0.4])]
    poses, projections = [], []
    for offset, R, T in zip(x_offsets, rotations, translations):
        seen = None if offset is None else (R @ (truth + [offset, 0.0, 0.0]) + T)[None, :]
        poses.append(seen)
        projections.append(np.hstack([R, T[:, None]]))
    valid = [i for i, p in enumerate(poses) if p is not None]
    sigma = None if sigma is None else np.asarray(sigma, float)[:, None]
    return Views([None] * 4, poses, sigma, valid), projections, truth


def test_median_resists_an_outlier_view():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    # sigma is not read: the outlier view claims to be the most certain
    views, projections, truth = _four_views([-0.002, 0.0, 0.003, 0.5],
                                            sigma=[0.03, 0.03, 0.03, 0.01])
    # midway between the two middle readings, 0 and 3 mm
    np.testing.assert_allclose(reconstruct_3d(views, projections, weighting="median"),
                               [truth + [0.0015, 0.0, 0.0]], atol=1e-12)
    # the mean carries a quarter of the 0.5 m outlier
    mean = reconstruct_3d(views, projections, weighting="mean")
    assert abs(mean[0, 0] - truth[0]) > 0.12


def test_weighted_median_resists_a_confident_outlier_view():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    # weights 1, 1, 1, 2.25: half the total (2.625) is reached at the 3 mm reading
    views, projections, truth = _four_views([-0.002, 0.0, 0.003, 0.5],
                                            sigma=[0.03, 0.03, 0.03, 0.02])
    np.testing.assert_allclose(reconstruct_3d(views, projections, weighting="wmedian"),
                               [truth + [0.003, 0.0, 0.0]], atol=1e-12)
    invvar = reconstruct_3d(views, projections, weighting="invvar")
    assert abs(invvar[0, 0] - truth[0]) > 0.2


def test_weighted_median_follows_a_view_holding_most_of_the_weight():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    # weights 1, 1, 1, 9: the last view alone is more than half
    views, projections, truth = _four_views([-0.002, 0.0, 0.003, 0.5],
                                            sigma=[0.03, 0.03, 0.03, 0.01])
    np.testing.assert_allclose(reconstruct_3d(views, projections, weighting="wmedian"),
                               [truth + [0.5, 0.0, 0.0]], atol=1e-12)


def test_medians_ignore_missing_views_and_missing_sigma():
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    # view 1 saw nobody; three readings left: the middle one
    views, projections, truth = _four_views([-0.002, None, 0.003, 0.5], sigma=[0.03] * 4)
    for rule in ("median", "wmedian"):
        np.testing.assert_allclose(reconstruct_3d(views, projections, weighting=rule),
                                   [truth + [0.003, 0.0, 0.0]], atol=1e-12)
    # a view without sigma has no vote; weights 1, 1, 1 on -2, 0 and 3 mm
    views, projections, truth = _four_views([-0.002, 0.0, 0.003, 0.5],
                                            sigma=[0.03, 0.03, 0.03, np.nan])
    np.testing.assert_allclose(reconstruct_3d(views, projections, weighting="wmedian"),
                               [truth + [0.0, 0.0, 0.0]], atol=1e-12)
    # no sigma at all: the unweighted median
    views, projections, truth = _four_views([-0.002, 0.0, 0.003, 0.5])
    np.testing.assert_allclose(reconstruct_3d(views, projections, weighting="wmedian"),
                               [truth + [0.0015, 0.0, 0.0]], atol=1e-12)


def _fusion_error_mm(rule, noise_sd, reported_sd, frames=4000, landmarks=5):
    """RMS error per axis (mm) of a rule on Gaussian views of known noise."""
    from rtcosmik.nlf.nlf import Views
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    rng = np.random.default_rng(10)
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    truth = rng.uniform(-1.0, 1.0, (landmarks, 3)) + [0.0, 0.0, 3.0]
    sigma = np.repeat(np.asarray(reported_sd, float)[:, None], landmarks, axis=1)
    squared = []
    for _ in range(frames):
        poses = [truth + rng.normal(0.0, sd, truth.shape) for sd in noise_sd]
        views = Views([None] * 4, poses, sigma, [0, 1, 2, 3])
        squared.append((reconstruct_3d(views, [identity] * 4, weighting=rule) - truth) ** 2)
    return 1000.0 * np.sqrt(np.mean(squared))


def test_inverse_variance_beats_the_mean_when_sigma_is_truthful():
    noise = reported = [0.010, 0.010, 0.040, 0.040]
    # closed form: mean sqrt(2*10^2 + 2*40^2)/4 = 14.58 mm; inverse variance
    # 1/sqrt(2/10^2 + 2/40^2) = 6.86 mm
    assert _fusion_error_mm("mean", noise, reported) == pytest.approx(14.58, rel=0.03)
    assert _fusion_error_mm("invvar", noise, reported) == pytest.approx(6.86, rel=0.03)


def test_inverse_variance_does_not_beat_the_mean_when_sigma_is_uninformative():
    noise, reported = [0.020] * 4, [0.010, 0.010, 0.040, 0.040]
    # closed form: mean 20/sqrt(4) = 10 mm; weights 16, 16, 1, 1 over 34 give
    # 20*sqrt(2*16^2 + 2)/34 = 13.34 mm
    assert _fusion_error_mm("mean", noise, reported) == pytest.approx(10.0, rel=0.03)
    assert _fusion_error_mm("invvar", noise, reported) == pytest.approx(13.34, rel=0.03)


def test_medians_are_fusion_rules_of_the_settings():
    from rtcosmik.triangulation.triangulation import FUSION_RULES
    assert FUSION_RULES == ("mean", "invvar", "median", "wmedian")
