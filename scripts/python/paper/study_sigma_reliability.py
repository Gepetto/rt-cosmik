#!/usr/bin/env python3
"""How well NLF's sigma predicts each view's landmark error against the mocap markers.

Per trial, every view's landmarks are brought to the world frame and compared with
the mocap markers at the same instant (clock offset removed). A landmark is a
surface vertex and a marker sits on the skin nearby, so the error carries a
constant part that sigma cannot know. It is removed in two ways, next to the raw
error:

``trial``   per trial and landmark, the mean error vector over frames and views
            (world frame): what the fusion cannot remove. The reference reading.
``view``    per trial, landmark and view, that view's own mean error vector.
``depth``, ``lateral``  the ``trial`` error along and across the view's line of
            sight to the marker.

Reported per view and landmark: the rank correlation of sigma with error over
the frames of a trial ("over time"); per landmark: the rank correlation across
views within a frame ("across views": both centred on the frame's mean, which
is what weighting views relies on), how often the view sigma ranks best (worst)
is the best (worst), a calibration table by sigma bin, and the fused landmark's
error under each fusion rule.

    python3 scripts/python/paper/study_sigma_reliability.py \\
        --views /root/workspace/oof/views --out /root/workspace/oof/t10/reliability
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from rtcosmik.folds import PARTICIPANTS
from rtcosmik.paper import fusion_validation as fv

TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
ROBOT_TASKS = ("RobotPolishing", "RobotWelding")
WORKERS = 10
OFFSETS = ("raw", "trial", "view", "depth", "lateral")
RULES = ("mean", "invvar", "median", "wmedian")
#: Sigma bins of the calibration table, metres.
SIGMA_EDGES = (0.0, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.12, np.inf)
REGIONS = {"pelvis": ("RASI", "LASI", "RPSI", "LPSI"), "trunk": ("C7", "RSHO", "LSHO"),
           "elbow": ("RELB", "LELB", "RMELB", "LMELB"), "wrist": ("RWRI", "LWRI", "RMWRI", "LMWRI"),
           "knee": ("RKNE", "LKNE", "RMKNE", "LMKNE"), "ankle": ("RANK", "LANK", "RMANK", "LMANK"),
           "foot": ("R5MHD", "L5MHD", "RTOE", "LTOE", "RHEE", "LHEE")}


def mocap_markers(path):
    """{name: (frames, 3) metres} of a raw marker export."""
    with open(path) as handle:
        header = next(csv.reader(handle))
    table = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    return {name[:-len("_X[mm]")]: table[:, i:i + 3] * 1e-3
            for i, name in enumerate(header) if name.endswith("_X[mm]")}


def world_views(recording, cam_dir):
    """(F, C, M, 3) landmarks of every view in the world frame, camera centres, per-frame fusions."""
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.nlf.nlf import Views
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    cameras = [int(c) for c in recording["cameras"]]
    _, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = (np.asarray(v) for v in load_world_transformation(cam_dir, cameras[0]))
    poses = recording["poses3d"].astype(float)
    sigma = recording["uncertainties"].astype(float)
    world, centres = np.empty_like(poses), []
    for c, projection in enumerate(projections):
        R, T = np.asarray(projection, float)[:, :3], np.asarray(projection, float)[:, 3]
        world[:, c] = (poses[:, c] - T) @ R @ world_R.T + world_T
        centres.append(-T @ R @ world_R.T + world_T)
    fused = {rule: np.full(poses[:, 0].shape, np.nan) for rule in RULES}
    for frame in range(len(poses)):
        valid = [c for c in range(len(cameras)) if np.isfinite(poses[frame, c]).all()]
        if not valid:
            continue
        views = Views([None] * len(cameras),
                      [poses[frame, c] if c in valid else None for c in range(len(cameras))],
                      sigma[frame], valid)
        for rule in RULES:
            fused[rule][frame] = reconstruct_3d(views, projections, rule) @ world_R.T + world_T
    return cameras, world, np.asarray(centres), sigma, fused


def without_offset(error, kind, sight):
    """(N, C, 3) error vectors of one landmark with the constant part of ``kind`` removed.

    ``sight`` holds the unit vectors from each view to the marker.
    """
    if kind == "raw":
        return error
    if kind == "view":
        return error - error.mean(axis=0)
    error = error - error.mean(axis=(0, 1))
    if kind == "trial":
        return error
    along = (error * sight).sum(axis=2, keepdims=True) * sight
    return along if kind == "depth" else error - along


def analyse(job):
    views_root, dataset, participant, task = job
    from rtcosmik.config_loader import settings
    from rtcosmik.paper.scoring import clock_offset
    with np.load(views_root / participant / f"{task}.npz") as handle:
        recording = {key: handle[key] for key in ("cameras", "poses3d", "uncertainties")}
    cameras, world, centres, sigma, fused = world_views(recording, dataset / "cam_params" / participant)
    truth = mocap_markers(dataset / "mocap" / "aligned" / participant / task / "markers_trajectories.csv")
    offset = clock_offset(settings, participant, task)       # view frame i + offset is mocap frame i
    frames = min(len(world) - offset, len(next(iter(truth.values()))))
    trial = {"participant": participant, "task": task}

    cells, landmarks = [], []
    bins = {kind: np.zeros((len(SIGMA_EDGES) - 1, 4)) for kind in OFFSETS}
    for m, name in enumerate(settings.marker_names):
        if name not in truth:
            continue
        gap = world[offset:offset + frames, :, m] - truth[name][:frames, None, :]
        ok = np.isfinite(gap).all(axis=(1, 2)) & np.isfinite(sigma[offset:offset + frames, :, m]).all(axis=1)
        if ok.sum() < 100:
            continue
        gap, s = gap[ok], sigma[offset:offset + frames, :, m][ok]
        sight = truth[name][:frames][ok][:, None, :] - centres
        sight /= np.linalg.norm(sight, axis=2, keepdims=True)
        row = {**trial, "landmark": name, "frames": int(ok.sum())}
        per_view = [{**trial, "landmark": name, "view": camera, "frames": int(ok.sum()),
                     "sigma_mean_mm": 1e3 * s[:, c].mean()} for c, camera in enumerate(cameras)]
        cells += per_view
        for kind in OFFSETS:
            error = np.linalg.norm(without_offset(gap, kind, sight), axis=2)            # (N, C)
            over_time = fv.rank_correlation(s, error)
            for c, entry in enumerate(per_view):
                entry[f"rho_time_{kind}"] = over_time[c]
                entry[f"error_rms_mm_{kind}"] = 1e3 * np.sqrt((error[:, c] ** 2).mean())
            centred = lambda x: (x - x.mean(axis=1, keepdims=True)).ravel()
            row[f"rho_views_{kind}"] = float(fv.rank_correlation(centred(s), centred(error)))
            row[f"best_hit_{kind}"], row[f"worst_hit_{kind}"] = fv.view_choice(s, error)
            for b, entry in enumerate(fv.calibration(s, error, SIGMA_EDGES)):
                if entry["count"]:
                    bins[kind][b] += entry["count"] * np.array(
                        [1.0, entry["sigma_mean"], entry["error_rms"] ** 2, entry["error_mean"]])
        for rule in RULES:
            fused_gap = (fused[rule][offset:offset + frames, m] - truth[name][:frames])[ok]
            row[f"fused_raw_mm_{rule}"] = 1e3 * np.sqrt((fused_gap ** 2).sum(1).mean())
            fused_gap = fused_gap - fused_gap.mean(axis=0)
            row[f"fused_trial_mm_{rule}"] = 1e3 * np.sqrt((fused_gap ** 2).sum(1).mean())
        landmarks.append(row)
    calibration = [{**trial, "offset": kind, "sigma_from": SIGMA_EDGES[b], "sigma_to": SIGMA_EDGES[b + 1],
                    "count": int(total[0]), "sigma_sum": total[1], "error_sq_sum": total[2],
                    "error_sum": total[3]}
                   for kind in OFFSETS for b, total in enumerate(bins[kind])]
    return cells, landmarks, calibration


# --------------------------------------------------------------------------- summary

def by_participant(rows, key, tasks=None, where=None):
    """Mean and SD over participants of each participant's mean of ``key``; and their number."""
    people = {}
    for row in rows:
        if (tasks is None or row["task"] in tasks) and (where is None or where(row)):
            value = float(row[key])
            if np.isfinite(value):
                people.setdefault(row["participant"], []).append(value)
    means = np.array([np.mean(v) for v in people.values()])
    if not len(means):
        return np.nan, np.nan, 0
    return float(means.mean()), float(means.std(ddof=1)) if len(means) > 1 else np.nan, len(means)


def scopes():
    return (("all tasks", None), ("robot tasks", ROBOT_TASKS),
            ("other tasks", tuple(t for t in TASKS if t not in ROBOT_TASKS)))


def cell(rows, key, tasks=None, where=None, digits=2):
    mean, sd, _ = by_participant(rows, key, tasks, where)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def summary(cells, landmarks, calibration):
    lines = []
    people = sorted({r["participant"] for r in landmarks})
    lines += [f"Participants ({len(people)}): {' '.join(people)}. "
              f"Trials: {len({(r['participant'], r['task']) for r in landmarks})}. "
              "Mean ± SD over participants of each participant's mean over trials, landmarks and views.", ""]

    lines += ["### Rank correlation of sigma with error", "",
              "| Tasks | Error | View error RMS (mm) | Over time, per view | Across views, per frame | "
              "Best view found | Worst view found |", "|---|---|---|---|---|---|---|"]
    for label, tasks in scopes():
        for kind in OFFSETS:
            lines.append(f"| {label} | {kind} | {cell(cells, f'error_rms_mm_{kind}', tasks, None, 1)} | "
                         f"{cell(cells, f'rho_time_{kind}', tasks)} | "
                         f"{cell(landmarks, f'rho_views_{kind}', tasks)} | "
                         f"{cell(landmarks, f'best_hit_{kind}', tasks)} | "
                         f"{cell(landmarks, f'worst_hit_{kind}', tasks)} |")
    lines += ["", "Chance for the last two columns is 0.25.", ""]

    views = sorted({r["view"] for r in cells})
    lines += ["### Per view (offset removed per trial)", "",
              "| View | Sigma (mm) | Error RMS (mm) | Rho over time, all | robot | other |", "|---|---|---|---|---|---|"]
    for view in views:
        mine = lambda r, view=view: r["view"] == view
        lines.append(f"| {view} | {cell(cells, 'sigma_mean_mm', None, mine, 1)} | "
                     f"{cell(cells, 'error_rms_mm_trial', None, mine, 1)} | "
                     + " | ".join(cell(cells, "rho_time_trial", tasks, mine) for _, tasks in scopes()) + " |")

    lines += ["", "### Per landmark region (offset removed per trial)", "",
              "| Region | Sigma (mm) | View error RMS (mm) | Rho over time | Rho across views, all | robot | other | "
              f"Fused error {' / '.join(RULES)} (mm) |", "|---|---|---|---|---|---|---|---|"]
    for region, names in list(REGIONS.items()) + [("all", sum(REGIONS.values(), ()))]:
        mine = lambda r, names=names: r["landmark"] in names
        fused = " / ".join(f"{by_participant(landmarks, f'fused_trial_mm_{rule}', None, mine)[0]:.1f}"
                           for rule in RULES)
        lines.append(f"| {region} | {cell(cells, 'sigma_mean_mm', None, mine, 1)} | "
                     f"{cell(cells, 'error_rms_mm_trial', None, mine, 1)} | "
                     f"{cell(cells, 'rho_time_trial', None, mine)} | "
                     + " | ".join(cell(landmarks, "rho_views_trial", tasks, mine) for _, tasks in scopes())
                     + f" | {fused} |")

    lines += ["", "### Fused landmark error by rule (mm, RMS over frames, mean over landmarks)", "",
              "| Tasks | Offset removed | " + " | ".join(RULES) + " |", "|---|---|" + "---|" * len(RULES)]
    for label, tasks in scopes():
        for kind in ("raw", "trial"):
            lines.append(f"| {label} | {kind} | "
                         + " | ".join(cell(landmarks, f"fused_{kind}_mm_{rule}", tasks, None, 1)
                                      for rule in RULES) + " |")

    lines += ["", "### Calibration: view error by sigma bin (offset removed per trial; frames pooled)", "",
              "| Sigma bin (mm) | Share of samples, all | Mean error (mm), all | RMS, all | Share, robot | "
              "Mean error, robot | RMS, robot |", "|---|---|---|---|---|---|---|"]
    pooled = {}
    for label, tasks in scopes()[:2]:
        for row in calibration:
            if row["offset"] == "trial" and (tasks is None or row["task"] in tasks):
                key = (label, float(row["sigma_from"]))
                pooled[key] = pooled.get(key, 0) + np.array(
                    [float(row[k]) for k in ("count", "error_sum", "error_sq_sum")])
    totals = {label: sum(v[0] for (l, _), v in pooled.items() if l == label) for label, _ in scopes()[:2]}
    for b in range(len(SIGMA_EDGES) - 1):
        parts = []
        for label, _ in scopes()[:2]:
            count, error, square = pooled[(label, float(SIGMA_EDGES[b]))]
            parts += [f"{100 * count / totals[label]:.1f} %",
                      f"{1e3 * error / count:.1f}" if count else "", f"{1e3 * np.sqrt(square / count):.1f}" if count else ""]
        upper = "" if np.isinf(SIGMA_EDGES[b + 1]) else f"{1e3 * SIGMA_EDGES[b + 1]:.0f}"
        lines.append(f"| {1e3 * SIGMA_EDGES[b]:.0f} to {upper} | " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/root/workspace/COMFI"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = [(args.views, args.dataset, p, t) for p in PARTICIPANTS for t in TASKS
            if (args.views / p / f"{t}.npz").exists()]
    cells, landmarks, calibration = [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for job, (a, b, c) in zip(jobs, pool.map(analyse, jobs)):
            cells += a
            landmarks += b
            calibration += c
            print(f"  {job[2]}/{job[3]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "view_landmark.csv", cells)
    write(args.out / "landmark.csv", landmarks)
    write(args.out / "calibration.csv", calibration)
    text = summary(cells, landmarks, calibration)
    (args.out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
