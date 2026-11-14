"""Fusion validation: perturbations of recorded views, and how well sigma predicts error."""
import hashlib

import numpy as np

#: Landmarks an injected outlier moves together, by name prefix.
LIMBS = {"right_arm": ("RELB", "RMELB", "RWRI", "RMWRI"),
         "left_arm": ("LELB", "LMELB", "LWRI", "LMWRI"),
         "right_leg": ("RKNE", "RMKNE", "RANK", "RMANK", "RHEE", "RTOE", "R5MHD"),
         "left_leg": ("LKNE", "LMKNE", "LANK", "LMANK", "LHEE", "LTOE", "L5MHD")}


def seeded(*key):
    """A generator that depends only on ``key`` (trial, condition), not on the process."""
    digest = hashlib.sha256("/".join(str(k) for k in key).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def stretches(frames, rate, mean_length, rng):
    """Boolean (frames,) flags, on for a share ``rate`` of frames in runs of mean ``mean_length``."""
    flags = np.zeros(frames, bool)
    if rate <= 0:
        return flags
    if rate >= 1:
        return ~flags
    off_length = mean_length * (1.0 - rate) / rate
    on, start = rng.random() < rate, 0
    while start < frames:
        length = int(rng.geometric(1.0 / max(mean_length if on else off_length, 1.0)))
        flags[start:start + length] = on
        start, on = start + length, not on
    return flags


def mask_views(keypoints, poses, sigma, rate, mean_length, rng):
    """Copies of a recording with each view dropped, independently, for stretches of frames."""
    keypoints, poses, sigma = keypoints.copy(), poses.copy(), sigma.copy()
    for view in range(poses.shape[1]):
        gone = stretches(len(poses), rate, mean_length, rng)
        keypoints[gone, view], poses[gone, view], sigma[gone, view] = np.nan, np.nan, np.nan
    return keypoints, poses, sigma


def inject_outliers(poses, names, size, rate, mean_length, rng):
    """Copy of ``poses`` (F, C, M, 3) with one view's limbs displaced for stretches of frames.

    One view per recording; each stretch moves one limb by ``size`` metres in one
    direction. Sigma is left as recorded. Returns the poses and the view.
    """
    poses = poses.copy()
    view = int(rng.integers(poses.shape[1]))
    flags = stretches(len(poses), rate, mean_length, rng)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], flags, [False]]).astype(int)))
    limbs = [[names.index(n) for n in members if n in names] for members in LIMBS.values()]
    for start, stop in zip(edges[0::2], edges[1::2]):
        direction = rng.normal(size=3)
        limb = limbs[int(rng.integers(len(limbs)))]
        poses[start:stop, view, limb] += size * direction / np.linalg.norm(direction)
    return poses, view


def ranks(values):
    """Average ranks along the first axis."""
    from scipy.stats import rankdata
    return rankdata(values, axis=0)


def rank_correlation(a, b):
    """Spearman correlation of two (N, ...) arrays along the first axis; NaN where constant."""
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(0), rb - rb.mean(0)
    norm = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (ra * rb).sum(0) / norm, np.nan)


def calibration(sigma, error, edges):
    """Per sigma bin: count, mean sigma, mean error and RMS error."""
    sigma, error = np.ravel(sigma), np.ravel(error)
    which = np.clip(np.searchsorted(edges, sigma, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        e, s = error[which == b], sigma[which == b]
        rows.append({"count": int(e.size),
                     "sigma_mean": float(s.mean()) if e.size else np.nan,
                     "error_mean": float(e.mean()) if e.size else np.nan,
                     "error_rms": float(np.sqrt((e ** 2).mean())) if e.size else np.nan})
    return rows


def view_choice(sigma, error):
    """How often the view sigma ranks best (worst) has the smallest (largest) error.

    ``sigma`` and ``error`` are (F, C, ...); chance is 1 / C.
    """
    best = (sigma.argmin(1) == error.argmin(1)).mean(0)
    worst = (sigma.argmax(1) == error.argmax(1)).mean(0)
    return best, worst
