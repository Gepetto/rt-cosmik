"""Triangulation variants, to find one that resolves depth from 2D keypoints.

The weighted linear DLT the pipeline ships minimises an *algebraic* residual.
That residual is scaled by each view's projective depth, so views that see the
subject from far away contribute rows with a different scale than near ones, and
the solution is pulled along the viewing direction. With COMFI's rig -- two
stereo pairs facing each other 5.3 m apart -- the far pair is exactly the one
that should fix depth, so a depth-scaled bias is expensive.

Three candidates, cheapest first:

``uniform``    drop the confidence weighting, to test whether it is the cause:
               the far cameras score lower simply for being far, so weighting by
               confidence may be discarding the views depth needs.
``iterative``  the textbook fix. Re-solve with each camera's rows divided by its
               projective depth, which turns the algebraic residual into the
               geometric one after two or three passes.
``robust``     iterative, plus a Huber reweighting on each view's reprojection
               residual, so one bad 2D detection cannot drag the solve. The
               published pipeline had an outlier rejection stage; this one does
               not.
"""

import numpy as np

from rtcosmik.triangulation.triangulation import (_build_dlt_system,
                                                  weights_from_uncertainties)


def _solve(a):
    b = np.matmul(np.transpose(a, (0, 2, 1)), a)
    _, _, vh = np.linalg.svd(b, full_matrices=False)
    homogeneous = vh[:, -1, :]
    return homogeneous[:, :3] / homogeneous[:, 3:4]


def triangulate(points_cj2, projections, weights, method="dlt", iterations=3,
                huber=2.0, eps=1e-9):
    """Triangulate (C, J, 2) normalised points with the chosen method.

    Args:
        points_cj2: undistorted normalised coordinates, NaN where absent.
        projections: (C, 3, 4) reference-frame-to-camera matrices.
        weights: (C, J) per-view weights, already normalised per joint.
        method: "dlt", "uniform", "iterative" or "robust".
        iterations: refinement passes for the iterative and robust methods.
        huber: Huber threshold, in multiples of the median residual.

    Returns:
        (J, 3) points in the reference frame.
    """
    projections = np.asarray(projections, dtype=np.float64)
    if method == "uniform":
        weights = np.where(np.isfinite(weights) & (weights > 0), 1.0, 0.0)
    base = np.array(weights, dtype=np.float64, copy=True)

    points = _solve(_build_dlt_system(points_cj2, projections, base))
    if method in ("dlt", "uniform"):
        return points

    for _ in range(iterations):
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        # Projective depth of each joint in each view: the third row of P times X.
        depth = np.einsum("cd,jd->cj", projections[:, 2, :], homogeneous)
        scale = base / np.maximum(np.abs(depth), eps)

        if method == "robust":
            projected = np.einsum("crd,jd->cjr", projections, homogeneous)
            uv = projected[:, :, :2] / np.where(np.abs(projected[:, :, 2:3]) < eps,
                                                eps, projected[:, :, 2:3])
            residual = np.linalg.norm(uv - points_cj2, axis=2)
            finite = np.isfinite(residual) & (base > 0)
            if finite.any():
                median = np.nanmedian(np.where(finite, residual, np.nan), axis=0)
                threshold = huber * np.maximum(median, eps)
                # Huber: full weight inside the threshold, 1/r beyond it.
                factor = np.where(residual <= threshold, 1.0,
                                  threshold / np.maximum(residual, eps))
                scale = scale * np.nan_to_num(factor, nan=0.0)

        points = _solve(_build_dlt_system(points_cj2, projections, scale))
    return points


def prepare(keypoints_list, mtxs, dists, confidences, power=2.0, floor=1e-3):
    """Undistort keypoints and build per-view weights, as the pipeline does."""
    import cv2

    num_cams = len(keypoints_list)
    num_points = min(len(k) for k in keypoints_list)
    points = np.full((num_cams, num_points, 2), np.nan)
    for i, kp in enumerate(keypoints_list):
        undistorted = cv2.undistortPoints(
            np.asarray(kp, dtype=np.float64)[:num_points].reshape(-1, 1, 2),
            mtxs[i], np.asarray(dists[i]).reshape(-1, 1))
        points[i] = undistorted[:, 0, :]
    sigma = 1.0 / np.clip(confidences, floor, None)
    weights = weights_from_uncertainties(sigma, num_cams, num_points, power=power)
    weights[~np.isfinite(points).all(axis=2)] = 0.0
    return points, weights


def _ray_information(points, projections, weights, cams, eps=1e-9):
    """Fisher information of a 3D point from a subset of cameras.

    A pixel error displaces the estimate perpendicular to the viewing ray by an
    amount proportional to range, and not at all along the ray. So each camera
    contributes ``(I - d d^T) / range**2`` and contributes nothing along its own
    line of sight. Summed over cameras, the weakest eigenvector is the direction
    the rig cannot see -- for two cameras facing each other, the line joining
    them, which on COMFI's rig is within 6 to 9 degrees of camera 0's depth axis.

    Returns (J, 3, 3).
    """
    from rtcosmik.triangulation.triangulation import camera_centres

    centres = camera_centres(projections)[cams]
    d = points[None, :, :] - centres[:, None, :]
    ranges = np.linalg.norm(d, axis=2, keepdims=True)
    d = d / np.maximum(ranges, eps)
    w = weights[cams][:, :, None, None]
    outer = d[:, :, :, None] * d[:, :, None, :]
    perpendicular = np.eye(3)[None, None, :, :] - outer
    return (perpendicular * w / np.maximum(ranges[..., None] ** 2, eps)).sum(axis=0)


def triangulate_pair_fused(points_cj2, projections, weights, pairs=((0, 1), (2, 3)),
                           method="iterative", eps=1e-9):
    """Solve each camera pair separately, then fuse by inverse covariance.

    A joint solve over all four rays minimises a single algebraic residual. With
    two nearly opposed pairs that residual is dominated by the well-conditioned
    directions, and the weak axis is left to whatever the least squares happens
    to do with it. Solving each pair on its own keeps each estimate with the
    covariance it actually has, so the fusion can weight the weak axis properly
    instead of letting it be averaged away.

    Each pair still sees the weak axis poorly, but the two pairs' errors along it
    are independent, so fusing them is worth sqrt(2) there before the
    inverse-covariance weighting does anything else.
    """
    projections = np.asarray(projections, dtype=np.float64)
    estimates, informations = [], []
    for cams in pairs:
        cams = list(cams)
        if not np.any(weights[cams] > 0):
            continue
        sub = np.zeros_like(weights)
        sub[cams] = weights[cams]
        x = triangulate(points_cj2, projections, sub, method=method)
        estimates.append(x)
        informations.append(_ray_information(x, projections, weights, cams))
    if not estimates:
        return np.zeros((points_cj2.shape[1], 3))
    if len(estimates) == 1:
        return estimates[0]

    total = sum(informations)
    rhs = sum(np.einsum("jab,jb->ja", F, x) for F, x in zip(informations, estimates))
    trace = np.trace(total, axis1=1, axis2=2)[:, None, None]
    total = total + np.eye(3)[None] * np.maximum(trace, eps) * 1e-6
    return np.linalg.solve(total, rhs)
