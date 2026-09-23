#!/usr/bin/env python3
"""Multi-view fusion study: our inverse-variance fusion against Bragagnolo et al.

Bragagnolo, Terreran, Allegro and Ghidoni, "Multi-view Pose Fusion for
Occlusion-Aware 3D Human Pose Estimation" (ECCV Workshops 2024, arXiv
2408.15810) fuse the same kind of input we do -- the absolute metric 3D skeleton
a monocular network regresses in each view (MeTRAbs there, its successor NLF
here) -- but weight and refine it differently:

``invvar``      ours: each view's landmark is weighted by the inverse of the
                variance the network itself reports (w = sigma^-2).
``reproj``      theirs, Eq. (1)-(2): a view's landmark is weighted by the inverse
                of its mean reprojection error into *all* views,
                w_ij = 1 / e_ij, e_ij = mean_k || pi_k(P_ij) - p_jk ||^2.
``reproj_sym``  theirs in full, Eq. (5): those weights, then a per-frame
                refinement of the fused points that minimises the reprojection
                error over all views plus a left/right limb-length symmetry cost.
``mean``        control: no weights at all.

Only the fusion changes. Every variant replays the same recorded NLF output
(``sweep.py --views-cache``) through the same world transform, filter and
moving-horizon IK as the shipped pipeline, and is scored exactly like the other
studies (``ik_filter_studies.py``), so the comparison isolates the fusion rule.
The refinement follows the paper: residuals in pixels for reprojection and in
millimetres for symmetry, which is the balance implied by its Eq. (5) on a
millimetre-scale skeleton, solved by damped Gauss-Newton from the fused pose.

Per trial and variant: accuracy against the mocap reference, marker error,
jitter, and the per-frame cost of the fusion itself (ms) next to the IK's, which
is what decides whether a variant can run at camera rate.

    python3 scripts/python/paper/study_fusion.py --views results/campaign/views \\
        --output-dir output/campaign --out results/campaign/paper \\
        --participants 1012 1118 1508 1602 1847 2112
"""
import argparse
import csv
import logging
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
REFERENCE_TAG = "mocap_reference"
TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
FS = 40.0

#: variant -> (label, weighting, refine)
VARIANTS = OrderedDict([
    ("invvar", ("inverse variance (ours)", "invvar", False)),
    ("reproj", ("reprojection-error weights (Bragagnolo et al.)", "reproj", False)),
    ("reproj_sym", ("reprojection weights + symmetry refinement (Bragagnolo et al.)", "reproj", True)),
    ("mean", ("unweighted mean", "mean", False)),
])
#: Limb pairs the symmetry cost ties together, in the evaluation's marker names:
#: upper arm, lower arm, upper leg, lower leg (the pelvis markers stand in for
#: the hips, which this marker set does not carry).
SYMMETRY_PAIRS = ((("RSHO", "RELB"), ("LSHO", "LELB")), (("RELB", "RWRI"), ("LELB", "LWRI")),
                  (("RASI", "RKNE"), ("LASI", "LKNE")), (("RKNE", "RANK"), ("LKNE", "LANK")))
GAUSS_NEWTON_STEPS = 5
DAMPING = 1e-3
FIELDS = ["variant", "label", "participant", "task", "frames", "joint_rmse_deg", "upper_rmse_deg",
          "lower_rmse_deg", "trunk_rmse_deg", "marker_raw_mm", "marker_depth_mm", "shoulder_flip_pct",
          "lag_frames", "jitter_deg", "fuse_ms_p50", "fuse_ms_p95", "solve_ms_p50", "failed_frames"]


def tag(variant):
    return f"fusion_{variant}"


# --------------------------------------------------------------------------- fusion

def undistorted_pixels(keypoints, K, dist):
    """(C, J, 2) detections mapped into the ideal pinhole image of each camera."""
    import cv2
    out = np.full(keypoints.shape, np.nan, float)
    for c in range(len(keypoints)):
        if np.isfinite(keypoints[c]).all():
            out[c] = cv2.undistortPoints(keypoints[c].reshape(-1, 1, 2).astype(np.float64),
                                         K[c], np.asarray(dist[c]).reshape(-1, 1), P=K[c])[:, 0, :]
    return out


def project(points, K, R, t):
    """Pixels of reference-frame points in a camera, and the depth used."""
    cam = points @ R.T + t
    z = np.maximum(cam[:, 2:3], 1e-6)
    return (cam[:, :2] / z) @ K[:2, :2].T + K[:2, 2], cam, z


def fuse(poses_ref, sigma, pixels, K, R, t, weighting):
    """(J, 3) fused points in the reference frame, from each view's own estimate."""
    present = np.isfinite(poses_ref).all(axis=(1, 2))
    if not present.any():
        return None
    if weighting == "invvar":
        w = np.where(np.isfinite(sigma), 1.0 / np.maximum(sigma, 1e-3) ** 2, 0.0)
    elif weighting == "mean":
        w = np.ones(poses_ref.shape[:2])
    else:                                    # Bragagnolo et al., Eq. (2)
        seen = [k for k in range(len(pixels)) if np.isfinite(pixels[k]).all()]
        error = np.zeros(poses_ref.shape[:2])
        for i in range(len(poses_ref)):
            if not present[i]:
                continue
            for k in seen:
                uv, _, _ = project(poses_ref[i], K[k], R[k], t[k])
                error[i] += ((uv - pixels[k]) ** 2).sum(axis=1)
            error[i] /= max(len(seen), 1)
        w = np.where(error > 0, 1.0 / np.maximum(error, 1e-9), 0.0)
    w = np.where(present[:, None], w, 0.0)
    total = w.sum(axis=0)
    if not np.any(total > 0):
        return None
    fused = np.einsum("cj,cjk->jk", w, np.nan_to_num(poses_ref)) / np.maximum(total, 1e-12)[:, None]
    return fused


def refine(points, pixels, K, R, t, names, steps=GAUSS_NEWTON_STEPS):
    """Bragagnolo et al., Eq. (5): reprojection error over all views plus a
    left/right limb-length symmetry cost, by damped Gauss-Newton.

    Reprojection residuals are in pixels and symmetry residuals in millimetres,
    the balance their formulation implies for a millimetre-scale skeleton.
    """
    index = {n: i for i, n in enumerate(names)}
    pairs = [tuple(index[n] for n in (a[0], a[1], b[0], b[1])) for a, b in SYMMETRY_PAIRS
             if all(n in index for n in a + b)]
    seen = [k for k in range(len(pixels)) if np.isfinite(pixels[k]).all()]
    if not seen:
        return points
    x = points.copy()
    n = 3 * len(x)
    for _ in range(steps):
        H = np.zeros((n, n))
        g = np.zeros(n)
        for k in seen:
            uv, cam, z = project(x, K[k], R[k], t[k])
            residual = uv - pixels[k]                                  # (J, 2)
            fx, fy = K[k][0, 0], K[k][1, 1]
            for j in range(len(x)):
                d = np.array([[fx / z[j, 0], 0, -fx * cam[j, 0] / z[j, 0] ** 2],
                              [0, fy / z[j, 0], -fy * cam[j, 1] / z[j, 0] ** 2]]) @ R[k]
                s = slice(3 * j, 3 * j + 3)
                H[s, s] += d.T @ d
                g[s] += d.T @ residual[j]
        for a, b, c, e in pairs:
            left, right = x[a] - x[b], x[c] - x[e]
            la, lb = np.linalg.norm(left), np.linalg.norm(right)
            if la < 1e-6 or lb < 1e-6:
                continue
            residual = 1000.0 * (la - lb)                              # millimetres
            jac = np.zeros(n)
            jac[3 * a:3 * a + 3] = 1000.0 * left / la
            jac[3 * b:3 * b + 3] = -1000.0 * left / la
            jac[3 * c:3 * c + 3] = -1000.0 * right / lb
            jac[3 * e:3 * e + 3] = 1000.0 * right / lb
            H += np.outer(jac, jac)
            g += jac * residual
        H[np.diag_indices_from(H)] += DAMPING * np.maximum(np.diag(H), 1.0)
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        x = x + step.reshape(-1, 3)
    return x


def markers_of(views_path, cam_dir, variant):
    """Replay one trial's recording under one fusion rule: [(frame, world markers)],
    and the per-frame cost of the fusion in ms."""
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.config_loader import settings
    _, weighting, do_refine = VARIANTS[variant]
    # Read the recording once: an NpzFile decompresses the whole array on every
    # access, which would otherwise dominate both the replay and its timing.
    with np.load(views_path) as handle:
        data = {key: handle[key] for key in ("cameras", "keypoints", "poses3d", "uncertainties")}
    cameras = [int(c) for c in data["cameras"]]
    K, dist, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    R = [np.asarray(p, float)[:, :3] for p in projections]
    t = [np.asarray(p, float)[:, 3] for p in projections]
    world_R, world_T = (np.asarray(v) for v in load_world_transformation(cam_dir, cameras[0]))
    names = list(settings.marker_names)

    frames, cost = [], []
    for frame in range(len(data["keypoints"])):
        poses = data["poses3d"][frame].astype(float)
        # Each view regresses in its own camera frame; bring them to the reference one.
        in_ref = np.stack([(poses[i] - t[i]) @ R[i] for i in range(len(cameras))])
        pixels = undistorted_pixels(data["keypoints"][frame].astype(float), K, dist)
        started = time.perf_counter()
        fused = fuse(in_ref, data["uncertainties"][frame].astype(float), pixels, K, R, t, weighting)
        if fused is not None and do_refine:
            fused = refine(fused, pixels, K, R, t, names)
        cost.append((time.perf_counter() - started) * 1e3)
        if fused is None:
            continue
        frames.append((frame, fused @ world_R.T + world_T))
    return frames, np.asarray(cost)


# --------------------------------------------------------------------------- replay and score

def replay(job):
    runs_root, views_root, participant, task, variant = job
    import yaml
    from rtcosmik.config_loader import settings
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    frames, fuse_ms = markers_of(views_root / participant / f"{task}.npz",
                                 DATASET / "cam_params" / participant, variant)
    marker_filter = MarkerFilter(len(settings.marker_names), settings)
    meta = yaml.safe_load((DATASET / "metadata" / f"{participant}.yaml").read_text())
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    run_dir = runs_root / participant / task / tag(variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(run_dir), markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    solve_ms, failed, q_last = [], 0, None
    for index, (frame, p3d) in enumerate(frames):
        points = marker_filter(p3d)
        mks = dict(zip(settings.marker_names, points))
        started = time.perf_counter()
        try:
            q = solver.solve(mks)
            ok = np.all(np.isfinite(q))
        except Exception:
            q, ok = None, False
        if index:
            solve_ms.append((time.perf_counter() - started) * 1e3)
        if not ok:
            failed += 1
            if q_last is None:
                continue
            q = q_last
        q_last = q
        row = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = map(float, mks[name])
        saver.save_markers(row)
        saver.save_joint_angles(OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
    saver.close()
    solve_ms = np.asarray(solve_ms) if solve_ms else np.asarray([np.nan])
    return {"fuse_ms_p50": float(np.nanpercentile(fuse_ms, 50)),
            "fuse_ms_p95": float(np.nanpercentile(fuse_ms, 95)),
            "solve_ms_p50": float(np.nanpercentile(solve_ms, 50)), "failed_frames": failed}


def score(job):
    runs_root, participant, task, variant, timing = job
    import trial_metrics as tm
    import ik_filter_studies as studies
    ev = tm.load_eval()
    run_dir = runs_root / participant / task / tag(variant)
    dofs, trial = tm.trial_metrics(ev, run_dir, runs_root / participant / task / REFERENCE_TAG,
                                   DATASET / "mocap" / "aligned" / participant / task, participant)
    by_group = lambda g: float(np.mean([d["rmse_deg"] for d in dofs if d["group"] == g]))
    row = {"variant": variant, "label": VARIANTS[variant][0], "participant": participant, "task": task,
           "frames": trial["frames"], "joint_rmse_deg": trial["joint_rmse_deg"],
           "upper_rmse_deg": by_group("upper"), "lower_rmse_deg": by_group("lower"),
           "trunk_rmse_deg": by_group("trunk"), "marker_raw_mm": trial["marker_raw_mm"],
           "marker_depth_mm": trial["marker_depth_mm"], "shoulder_flip_pct": trial["shoulder_flip_pct"],
           "lag_frames": trial["lag_frames"]}
    row["jitter_deg"] = studies.smoothness_and_limits(run_dir, [d["dof"] + "[rad]" for d in dofs])["jitter_deg"]
    row.update(timing)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, default=REPO / "results" / "campaign" / "views")
    ap.add_argument("--output-dir", type=Path, default=REPO / "output" / "campaign")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "campaign" / "paper")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    people = args.participants or sorted(p.name for p in args.views.iterdir() if p.is_dir())
    trials = [(p, t) for p in people for t in TASKS if (args.views / p / f"{t}.npz").exists()]
    jobs = [(args.output_dir, args.views, p, t, v) for v in args.variants for p, t in trials]
    print(f"{len(trials)} trials x {len(args.variants)} variants", flush=True)

    timings = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, timing in zip(jobs, pool.map(replay, jobs)):
            timings[(job[2], job[3], job[4])] = timing
            print(f"  replayed {job[4]} {job[2]}/{job[3]}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        score_jobs = [(args.output_dir, p, t, v, timings[(p, t, v)]) for v in args.variants for p, t in trials]
        for row in pool.map(score, score_jobs):
            rows.append(row)
            print(f"  scored {row['variant']} {row['participant']}/{row['task']}: "
                  f"{row['joint_rmse_deg']:.2f} deg", flush=True)

    out = args.out / "studies"
    out.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        with open(out / f"fusion_{variant}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([r for r in rows if r["variant"] == variant])
    print(f"rows -> {out}/fusion_*.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
