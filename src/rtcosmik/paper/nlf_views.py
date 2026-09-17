"""Record NLF's per-camera output during a sweep, and rebuild markers from it later.

NLF estimates each image on its own, so one 4-camera run already holds what a
1- or 2-camera run of the same trial would see: each camera's 2D keypoints,
metric 3D pose and uncertainty. Recording them lets every configuration that
only differs after NLF -- fewer cameras, 2D triangulation instead of 3D fusion,
another IK or filter -- be replayed on the CPU instead of re-running the GPU.

Replay rebuilds world-frame markers exactly as ``sweep.run_nlf`` does. From the
same recording it reproduces the 4-camera run bit for bit; a subset of cameras
differs from a direct run of that subset only through the detector batch
(0.3-0.4 mm median marker difference, at most 0.08 deg of joint RMSE per trial,
measured on all six trials of participant 1012).
"""
from pathlib import Path

import numpy as np


class ViewsRecorder:
    """Accumulate one trial's per-camera NLF output, frame by frame."""

    def __init__(self, cameras, markers):
        self.cameras, self.markers = list(cameras), markers
        self.keypoints, self.poses3d, self.uncertainties = [], [], []

    def append(self, views):
        """One frame's :class:`~rtcosmik.nlf.nlf.Views`; NaN where a camera saw no one."""
        C, M = len(self.cameras), self.markers
        keypoints = np.full((C, M, 2), np.nan, np.float32)
        poses = np.full((C, M, 3), np.nan, np.float32)
        sigma = np.full((C, M), np.nan, np.float32)
        for c in views.valid_cam_ids:
            keypoints[c] = views.keypoints[c]
            if views.poses3d[c] is not None:
                poses[c] = views.poses3d[c]
        if views.uncertainties is not None:
            sigma[:] = views.uncertainties
        self.keypoints.append(keypoints)
        self.poses3d.append(poses)
        self.uncertainties.append(sigma)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, cameras=np.asarray(self.cameras),
                            keypoints=np.asarray(self.keypoints, dtype=np.float32),
                            poses3d=np.asarray(self.poses3d, dtype=np.float32),
                            uncertainties=np.asarray(self.uncertainties, dtype=np.float32))


def replay_markers(path, cam_dir, cameras=None, reconstruction="fuse3d"):
    """World-frame markers per frame, rebuilt from a recording.

    ``cameras`` selects a subset of the recorded cameras (default: all, in the
    recorded order; the first is the reference frame). ``reconstruction`` is
    "fuse3d" (the shipped path) or "tri2d" (weighted DLT of NLF's 2D keypoints).
    Returns a list of ``(frame_index, (M, 3) array)`` for the frames that
    produced markers, as ``run_nlf`` would have kept them.
    """
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.nlf.nlf import Views
    from rtcosmik.triangulation.triangulation import reconstruct_3d, triangulate_points

    data = np.load(path)
    recorded = [int(c) for c in data["cameras"]]
    cameras = recorded if cameras is None else list(cameras)
    rows = [recorded.index(c) for c in cameras]
    K, P, S = data["keypoints"][:, rows], data["poses3d"][:, rows], data["uncertainties"][:, rows]
    mtxs, dists, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = load_world_transformation(cam_dir, cameras[0])
    world_R, world_T = np.asarray(world_R), np.asarray(world_T)

    frames = []
    for frame in range(len(K)):
        valid = [c for c in range(len(cameras)) if np.isfinite(K[frame, c]).all()]
        keypoints = [K[frame, c] if c in valid else None for c in range(len(cameras))]
        poses = [P[frame, c] if c in valid and np.isfinite(P[frame, c]).all() else None
                 for c in range(len(cameras))]
        sigma = S[frame].astype(np.float64)
        views = Views(keypoints, poses, None if not valid or np.isnan(sigma).all() else sigma, valid)
        if reconstruction == "tri2d":
            if any(k is None for k in keypoints) or len(cameras) < 2:
                continue
            p3d = triangulate_points(keypoints, mtxs, dists, projections,
                                     uncertainties=views.uncertainties)
        else:
            p3d = reconstruct_3d(views, projections)
        if len(p3d) == 0:
            continue
        frames.append((frame, np.asarray(p3d) @ world_R.T + world_T))
    return frames
