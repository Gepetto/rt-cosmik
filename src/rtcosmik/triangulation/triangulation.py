import numpy as np
import cv2

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

def _build_dlt_system(points_cj2: np.ndarray, projections_arr: np.ndarray,
                      weights: np.ndarray = None) -> np.ndarray:
    """Build batched DLT linear systems with shape (J, 2C, 4).

    ``weights`` is an optional (C, J) array scaling each camera's two rows for
    each joint. Scaling a camera's rows by w is a weighted least-squares on the
    algebraic DLT residual: w = 0 is exactly equivalent to leaving that camera
    out of the system, and uniform weights reproduce the unweighted solve.
    """
    if projections_arr.ndim != 3 or projections_arr.shape[1:] != (3, 4):
        raise ValueError("projections must be an array-like of shape (C, 3, 4)")

    num_cams = points_cj2.shape[0]
    if projections_arr.shape[0] != num_cams:
        raise ValueError("Number of projections must match number of camera observations")

    x = points_cj2[:, :, 0]
    y = points_cj2[:, :, 1]
    p0 = projections_arr[:, 0, :]
    p1 = projections_arr[:, 1, :]
    p2 = projections_arr[:, 2, :]

    a0 = y[:, :, None] * p2[:, None, :] - p1[:, None, :]
    a1 = p0[:, None, :] - x[:, :, None] * p2[:, None, :]

    if weights is not None:
        scale = np.asarray(weights, dtype=np.float64)[:, :, None]
        a0 = a0 * scale
        a1 = a1 * scale
        # A camera with no detection contributes NaN rows; zero weight is meant
        # to drop it, so clear the NaNs the multiply leaves behind (0 * NaN).
        a0 = np.nan_to_num(a0, nan=0.0, posinf=0.0, neginf=0.0)
        a1 = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)

    num_points = points_cj2.shape[1]
    a = np.empty((num_points, 2 * num_cams, 4), dtype=np.float64)
    a[:, 0::2, :] = np.transpose(a0, (1, 0, 2))
    a[:, 1::2, :] = np.transpose(a1, (1, 0, 2))
    return a


def triangulate_points_torch(points_cj2_torch, projections_torch, return_numpy: bool = True):
    """
    Triangulate from undistorted normalized coordinates entirely in torch.

    Args:
        points_cj2_torch: torch.Tensor with shape (C, J, 2).
        projections_torch: torch.Tensor with shape (C, 3, 4) on the same device.
        return_numpy: if True, returns np.ndarray (J, 3), else torch.Tensor (J, 3).
    """
    import torch

    if points_cj2_torch.ndim != 3 or points_cj2_torch.shape[-1] != 2:
        raise ValueError("points_cj2_torch must have shape (C, J, 2)")
    if projections_torch.ndim != 3 or projections_torch.shape[1:] != (3, 4):
        raise ValueError("projections_torch must have shape (C, 3, 4)")
    if points_cj2_torch.shape[0] != projections_torch.shape[0]:
        raise ValueError("Number of cameras in points and projections must match")

    x = points_cj2_torch[:, :, 0]
    y = points_cj2_torch[:, :, 1]
    p0 = projections_torch[:, 0, :]
    p1 = projections_torch[:, 1, :]
    p2 = projections_torch[:, 2, :]

    a0 = y[..., None] * p2[:, None, :] - p1[:, None, :]
    a1 = p0[:, None, :] - x[..., None] * p2[:, None, :]

    num_points = points_cj2_torch.shape[1]
    num_cams = points_cj2_torch.shape[0]
    a = torch.empty((num_points, 2 * num_cams, 4), dtype=points_cj2_torch.dtype, device=points_cj2_torch.device)
    a[:, 0::2, :] = a0.permute(1, 0, 2)
    a[:, 1::2, :] = a1.permute(1, 0, 2)

    b = torch.matmul(a.transpose(-2, -1), a)
    _, _, vh = torch.linalg.svd(b, full_matrices=False)
    homog = vh[:, -1, :]
    xyz = homog[:, :3] / homog[:, 3:4]
    if return_numpy:
        return xyz.detach().cpu().numpy()
    return xyz


def camera_centres(projections):
    """Camera centres in the reference frame, from ``[R | T]`` projections.

    A projection maps a reference-frame point into the camera, ``x = R p + T``,
    so the camera sits at ``-R.T @ T``.

    Returns:
        np.ndarray: (C, 3) centres.
    """
    centres = []
    for projection in projections:
        matrix = np.asarray(projection, dtype=np.float64)
        rotation, translation = matrix[:, :3], matrix[:, 3]
        centres.append(-rotation.T @ translation)
    return np.asarray(centres, dtype=np.float64)


def distance_scaled_uncertainties(uncertainties, points3d, centres, floor=0.1):
    """Fold range into per-camera uncertainty: ``sigma_cj *= distance(c, j)``.

    A triangulated point's depth error grows with range -- roughly
    ``d**2 / (baseline * f)`` for a stereo pair -- so how far a joint is from a
    camera predicts that camera's reliability for it, independently of how
    confident the detector is. Confidence alone misses this: a detector can be
    perfectly sure about a joint it sees from 6 m away, and still localise it
    worse than a less certain view from 2 m.

    Taking sigma proportional to distance is the principled form and, unlike a
    Gaussian proximity kernel, introduces no length scale to tune. Weights are
    normalised per joint downstream, so only the ratio between cameras matters.

    The caller supplies ``points3d`` from the *previous* frame, which keeps this
    causal -- no peeking at the solve it is about to weight.

    MEASURED AND NOT ADOPTED. On 10 COMFI trials this is a wash where the cameras
    are roughly equidistant and a disaster where they are not: StraightWalking,
    with a 2.61x near/far range ratio, went from 24.3 to 39.5 deg. The reason is
    that proximity is the wrong prior for a multi-view DLT. What conditions the
    solve is angular diversity, and the distant views are usually the wide-
    parallax ones, so downweighting by range strips out exactly the geometry the
    triangulation depends on -- effective cameras per joint fell 3.39 to 3.16 on
    that trial.

    The idea was taken from a pipeline that fused two already-triangulated stereo
    estimates, and there it is sound: each pair has already spent its own
    baseline, so proximity really does predict which pair to trust. It does not
    survive the move to fusing 2D rays. Kept, off by default, because the
    negative result is worth more than the code.

    Args:
        uncertainties: (C, J) per-camera per-joint sigma, or None for distance
            weighting alone.
        points3d: (J, 3) reference-frame points, from the previous frame.
        centres: (C, 3) camera centres, from :func:`camera_centres`.
        floor: metres; distances below this are clamped, so a joint that lands
            near a camera centre cannot produce a zero uncertainty.

    Returns:
        np.ndarray: (C, J) scaled uncertainties.
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    distances = np.linalg.norm(points3d[None, :, :] - centres[:, None, :], axis=2)
    distances = np.maximum(distances, floor)
    if uncertainties is None:
        return distances
    return np.asarray(uncertainties, dtype=np.float64) * distances


def weights_from_uncertainties(uncertainties, num_cams, num_points,
                               power=2.0, eps=1e-3):
    """Turn NLF's per-joint uncertainties into per-camera, per-joint DLT weights.

    NLF reports a spread for every joint in every view, and it rises sharply when
    that joint is occluded or out of frame. Weighting each camera's contribution
    by ``1 / sigma**power`` is inverse-variance weighting: views that can see a
    joint well dominate its solve, while a view whose arm is hidden behind the
    torso is suppressed for *that joint only* and still contributes its good
    observations of every other joint.

    ``power=2`` is plain inverse-variance and needs no threshold to tune. Higher
    powers keep improving the random error slightly but concentrate the estimate
    on a single camera, which drags the result toward that camera's landmark
    convention, so the bias grows.

    Weights are normalised per joint so the largest is 1, which keeps the DLT
    system's scaling independent of the absolute uncertainty units. Cameras with
    a missing or non-finite uncertainty get weight 0.

    Returns:
        np.ndarray: (num_cams, num_points) weights in [0, 1].
    """
    if uncertainties is None:
        return np.ones((num_cams, num_points), dtype=np.float64)

    sigma = np.asarray(uncertainties, dtype=np.float64)
    if sigma.shape != (num_cams, num_points):
        raise ValueError(
            f"uncertainties must have shape {(num_cams, num_points)}, got {sigma.shape}")

    valid = np.isfinite(sigma) & (sigma > 0.0)
    weights = np.zeros_like(sigma)
    np.divide(1.0, np.maximum(sigma, eps) ** power, out=weights, where=valid)

    peak = weights.max(axis=0, keepdims=True)
    np.divide(weights, peak, out=weights, where=peak > 0.0)

    # A joint no camera reported an uncertainty for falls back to a plain solve
    # over whatever views did see it, rather than yielding an empty system.
    dead = ~(peak > 0.0)[0]
    if dead.any():
        weights[:, dead] = 1.0
    return weights


def triangulate_points(keypoints_list, mtxs, dists, projections, uncertainties=None,
                       power=2.0, return_valid=False):
    """
    Triangulate 3D points from multi-view 2D keypoints.

    Cameras that lost their detection may be passed as None; they are dropped
    from the solve instead of raising. When ``uncertainties`` is given (an
    (C, J) array from NLF), each camera's contribution to each joint is weighted
    by 1/sigma**power, so an occluded limb in one view stops corrupting that
    limb's 3D estimate while that view still helps everywhere else. Without it
    the solve is the plain unweighted DLT.

    All operations run on CPU with vectorized NumPy.

    Args:
        keypoints_list: per-camera (J, 2) pixel keypoints, or None for a camera
            with no detection this frame.
        mtxs, dists: per-camera intrinsics and distortion coefficients.
        projections: per-camera 3x4 [R | T] (no intrinsics: points are
            undistorted to normalized coordinates first).
        uncertainties: optional (C, J) per-joint spread from NLF.
        power: exponent of the inverse-uncertainty weighting.
        return_valid: also return, per joint, how many cameras meaningfully
            contributed (weight above 1e-6 of the best view for that joint).

    Returns:
        np.ndarray: (J, 3) triangulated points, or ``(points, valid_counts)``
        when ``return_valid`` is set. Joints seen by fewer than two cameras are
        still returned (solved from what is available) so the output keeps a row
        per marker; use ``return_valid`` to identify them.
    """
    num_cams = len(keypoints_list)
    present = [k is not None and len(k) > 0 for k in keypoints_list]
    if not any(present):
        return (np.zeros((0, 3)), np.zeros(0, dtype=int)) if return_valid else np.zeros((0, 3))

    num_points = min(len(k) for k, ok in zip(keypoints_list, present) if ok)
    if num_points == 0:
        return (np.zeros((0, 3)), np.zeros(0, dtype=int)) if return_valid else np.zeros((0, 3))

    points_cj2 = np.full((num_cams, num_points, 2), np.nan, dtype=np.float64)
    for ii, ok in enumerate(present):
        if not ok:
            continue
        pts = np.asarray(keypoints_list[ii], dtype=np.float64)[:num_points]
        undistorted = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), mtxs[ii], np.asarray(dists[ii]).reshape(-1, 1))
        points_cj2[ii] = undistorted[:, 0, :]

    weights = weights_from_uncertainties(
        uncertainties, num_cams, num_points, power=power)
    # A camera without a detection contributes nothing, whatever its weight.
    weights[~np.asarray(present)] = 0.0
    weights[~np.isfinite(points_cj2).all(axis=2)] = 0.0

    projections_arr = np.asarray(projections, dtype=np.float64)
    a = _build_dlt_system(points_cj2, projections_arr, weights)
    b = np.matmul(np.transpose(a, (0, 2, 1)), a)
    _, _, vh = np.linalg.svd(b, full_matrices=False)
    homog = vh[:, -1, :]
    points = homog[:, :3] / homog[:, 3:4]

    if return_valid:
        # Weights are normalised to a per-joint peak of 1, so a camera below
        # this share of the best view is not meaningfully contributing.
        return points, (weights > 1e-6).sum(axis=0)
    return points


def fuse_camera_poses3d(poses3d_list, projections, uncertainties=None,
                        power=2.0, eps=1e-3):
    """Fuse each camera's monocular 3D pose into one estimate in the reference frame.

    An alternative to triangulating 2D keypoints. NLF regresses a metric 3D pose
    per view, and that pose is anatomically coherent: it comes from a body model,
    so its segment lengths stay near-constant. Triangulating the 43 markers
    independently throws that away -- nothing ties the markers together, so bone
    lengths wobble frame to frame, and many NLF landmarks are *internal* points
    (pelvis, hip centres, spine) that no camera truly observes, so each view
    guesses them differently and triangulation compounds the disagreement.

    Averaging the per-view 3D poses instead keeps each skeleton's coherence while
    still cancelling the depth bias that makes any single view inaccurate.

    Each view's pose is expressed in its own camera frame, so it is first mapped
    into the reference camera frame using that camera's ``[R | T]``, which takes
    reference-frame points into camera coordinates: ``p_ref = R.T @ (p_cam - T)``.

    Args:
        poses3d_list: per-camera (J, 3) poses in that camera's frame, or None
            for a camera with no detection.
        projections: per-camera 3x4 ``[R | T]`` mapping reference frame to camera.
        uncertainties: optional (C, J) per-joint spread from NLF; views are
            combined by inverse-variance weights when given, else by plain mean.
        power: exponent of the inverse-uncertainty weighting.
        eps: floor on sigma, guarding against a zero-uncertainty view dominating.

    Returns:
        np.ndarray: (J, 3) fused pose in the reference camera frame.
    """
    num_cams = len(poses3d_list)
    present = [p is not None and len(p) > 0 for p in poses3d_list]
    if not any(present):
        return np.zeros((0, 3))

    num_points = min(len(p) for p, ok in zip(poses3d_list, present) if ok)
    if num_points == 0:
        return np.zeros((0, 3))

    in_ref = np.full((num_cams, num_points, 3), np.nan, dtype=np.float64)
    for ii, ok in enumerate(present):
        if not ok:
            continue
        pose = np.asarray(poses3d_list[ii], dtype=np.float64)[:num_points]
        rotation = np.asarray(projections[ii], dtype=np.float64)[:, :3]
        translation = np.asarray(projections[ii], dtype=np.float64)[:, 3]
        in_ref[ii] = (pose - translation) @ rotation

    weights = weights_from_uncertainties(
        uncertainties, num_cams, num_points, power=power, eps=eps)
    weights[~np.asarray(present)] = 0.0
    weights[~np.isfinite(in_ref).all(axis=2)] = 0.0

    total = weights.sum(axis=0, keepdims=True)
    # A joint no view could contribute falls back to a plain mean of whatever
    # finite values exist, so the output always keeps one row per marker.
    dead = (total <= 0.0)[0]
    if dead.any():
        weights[:, dead] = np.isfinite(in_ref[:, dead]).all(axis=2)
        total = weights.sum(axis=0, keepdims=True)

    fused = np.nansum(np.nan_to_num(in_ref) * weights[:, :, None], axis=0)
    return fused / np.maximum(total.T, 1e-12)


def reconstruct_3d(views, projections, power=2.0):
    """Turn one frame of per-camera observations into 3D points in the reference frame.

    Views are combined by fusing the metric 3D pose each one regresses, weighted
    by inverse variance. That keeps each view's body-model coherence, so segment
    lengths stay stable, while combining views cancels the depth bias a single
    view has. It works with any number of cameras, one included.

    Args:
        views: a :class:`~rtcosmik.nlf.nlf.Views` for this frame.
        projections: per-camera 3x4 ``[R | T]`` mapping reference frame to camera.
        power: exponent of the inverse-uncertainty weighting.

    Returns:
        np.ndarray: (J, 3) points in the reference camera frame, empty when no
        view saw the subject.
    """
    if not views.valid_cam_ids:
        return np.zeros((0, 3))
    return fuse_camera_poses3d(
        views.poses3d, projections, uncertainties=views.uncertainties, power=power)
