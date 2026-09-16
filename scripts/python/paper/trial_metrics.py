#!/usr/bin/env python3
"""Extract every per-trial metric the paper needs, into tidy tables.

For each arm and each trial it scored, after aligning the run to the mocap
reference with the knee-flexion cross-correlation lag (the protocol used
throughout the study):

``per_dof/<arm>.csv`` -- one row per scored joint angle:
    RMSE, MAE, bias (mean signed error), SD of the error (the RMSE left once
    the constant offset is removed; RMSE^2 = bias^2 + SD^2 exactly), Pearson r.

``per_trial/<arm>.csv`` -- one row per trial:
    lag (frames), frames scored, mean joint RMSE over the scored DoF, the share
    of frames in which a shoulder sits in the wrong configuration (any shoulder
    DoF more than ``FLIP_DEG`` off the reference: with the arm overhead, flexion
    reaches the model's +-180 deg limit and the IK can settle on the equivalent
    pose through abduction and rotation instead),
    free-flyer position RMS (mm) and orientation error (deg), and marker error
    against the raw Vicon markers split two ways, RMS in mm: depth / lateral
    along camera 0's optical axis, and whole-body translation / shape.

Nothing is averaged here. Aggregation over participants, the statistics and the
paper tables are built on top of these files, so every number in the paper can
be traced back to one row.

    python3 scripts/python/paper/trial_metrics.py --arms nlf_0 fastsam_0
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
REFERENCE_TAG = "mocap_reference"
LOWER, UPPER, TRUNK = ("Hip", "Knee", "Ankle"), ("Clavicle", "Shoulder", "Elbow"), ("Lumbar", "Cervical")
FACE = {"Nose", "Head", "REar", "LEar", "REye", "LEye"}
FLIP_DEG = 90.0

DOF_FIELDS = ["arm", "participant", "task", "dof", "group", "rmse_deg", "mae_deg",
              "bias_deg", "sd_deg", "r", "frames"]
TRIAL_FIELDS = ["arm", "participant", "task", "lag_frames", "frames", "scored_dof",
                "joint_rmse_deg", "shoulder_flip_pct", "freeflyer_pos_mm", "freeflyer_rot_deg",
                "marker_raw_mm", "marker_depth_mm", "marker_lateral_mm",
                "marker_translation_mm", "marker_shape_mm", "marker_count"]


def load_eval():
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_of(name):
    for keys, group in ((LOWER, "lower"), (UPPER, "upper"), (TRUNK, "trunk")):
        if any(k in name for k in keys):
            return group
    return None


def camera0_axis(participant, cache={}):
    if participant not in cache:
        from rtcosmik.camera.cam_utils import load_world_transformation
        R, _ = load_world_transformation(DATASET / "cam_params" / participant, 0)
        cache[participant] = np.asarray(R) @ np.array([0.0, 0.0, 1.0])
    return cache[participant]


def trial_metrics(ev, run_dir, ref_dir, truth_dir, participant, lag=None):
    """All metrics for one trial. Returns (dof_rows, trial_row).

    ``lag`` forces the alignment (e.g. 0 to score without compensation); by
    default it is estimated by knee-flexion cross-correlation.
    """
    run, ref = ev.load_run(str(run_dir)), ev.load_run(str(ref_dir))
    if lag is None:
        lag, _ = ev.estimate_lag(run, ref)
    a, b = ev.apply_lag(run, ref, lag)
    joints, freeflyer = ev.compare_joint_angles(a, b)
    locked = {j["name"] for j in joints if j.get("locked")}

    header, A, B = a["joint_header"], a["joint_values"], b["joint_values"]
    n = min(len(A), len(B))
    dof_rows, shoulder = [], []
    for i, name in enumerate(header):
        if not name.endswith("[rad]") or name in locked or group_of(name) is None:
            continue
        x, y = A[:n, i], B[:n, i]
        ok = np.isfinite(x) & np.isfinite(y)
        d = (np.degrees(x[ok] - y[ok]) + 180.0) % 360.0 - 180.0
        if d.size < 3:
            continue
        if "Shoulder" in name:
            full = np.zeros(n, dtype=bool)
            full[ok] = np.abs(d) > FLIP_DEG
            shoulder.append(full)
        r = (float(np.corrcoef(x[ok], y[ok])[0, 1])
             if x[ok].std() > 1e-9 and y[ok].std() > 1e-9 else np.nan)
        dof_rows.append({"dof": name.replace("[rad]", ""), "group": group_of(name),
                         "rmse_deg": float(np.sqrt((d ** 2).mean())),
                         "mae_deg": float(np.abs(d).mean()),
                         "bias_deg": float(d.mean()), "sd_deg": float(d.std()),
                         "r": r, "frames": int(d.size)})

    linear = {j["name"]: j["rmse"] for j in joints if j["unit"] == "m"}
    position = [linear[k] for k in linear if k.startswith("Freeflyer")]
    trial = {
        "lag_frames": int(lag), "frames": n, "scored_dof": len(dof_rows),
        "joint_rmse_deg": float(np.mean([row["rmse_deg"] for row in dof_rows])),
        "shoulder_flip_pct": (float(100 * np.any(shoulder, axis=0).mean())
                              if shoulder else np.nan),
        "freeflyer_pos_mm": (float(np.sqrt(np.sum(np.square(position)))) * 1000
                             if len(position) == 3 else np.nan),
        "freeflyer_rot_deg": float(freeflyer.get("orientation_deg", np.nan)),
    }

    # Markers against the raw Vicon trajectories, same lag. Facial markers have
    # no Vicon counterpart, and mocap_reference's Head/ears are band-derived
    # stand-ins, so only markers present in the raw export are compared.
    truth = ev.load_run(str(truth_dir))
    names = sorted((set(run["markers"]) & set(truth["markers"])) - FACE)
    trial.update({"marker_raw_mm": np.nan, "marker_depth_mm": np.nan,
                  "marker_lateral_mm": np.nan, "marker_translation_mm": np.nan,
                  "marker_shape_mm": np.nan, "marker_count": len(names)})
    if names:
        start, ref_start = max(0, lag), max(0, -lag)
        P = np.stack([run["markers"][m][start:] for m in names], 1)
        Q = np.stack([truth["markers"][m][ref_start:] for m in names], 1)
        k = min(len(P), len(Q))
        e = P[:k] - Q[:k]
        e = e[np.isfinite(e).all(2).all(1)]
        if len(e):
            axis = camera0_axis(participant)
            along = e @ axis
            translation = e.mean(axis=1, keepdims=True)
            rms = lambda v: float(np.sqrt((v ** 2).sum(-1).mean()) * 1000)
            trial.update({
                "marker_raw_mm": rms(e),
                "marker_depth_mm": float(np.sqrt((along ** 2).mean()) * 1000),
                "marker_lateral_mm": rms(e - along[..., None] * axis),
                "marker_translation_mm": rms(translation),
                "marker_shape_mm": rms(e - translation)})
    return dof_rows, trial


def trials_for(results, arm):
    """Trials an arm completed, from its rescored summary."""
    path = results / "vs_mocap" / f"{arm}.csv"
    if not path.exists():
        return []
    return [(r["participant"], r["task"]) for r in csv.DictReader(open(path))
            if r.get("status") == "ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="where run folders live; defaults to settings.output_dir")
    ap.add_argument("--results", type=Path, default=REPO / "results",
                    help="folder whose vs_mocap/ lists the trials each arm completed")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "paper")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    ev = load_eval()
    runs_root = args.output_dir or Path(settings.output_dir)

    for arm in args.arms:
        trials = trials_for(args.results, arm)
        dof_out = args.out / "per_dof" / f"{arm}.csv"
        trial_out = args.out / "per_trial" / f"{arm}.csv"
        dof_out.parent.mkdir(parents=True, exist_ok=True)
        trial_out.parent.mkdir(parents=True, exist_ok=True)
        failed = 0
        with open(dof_out, "w", newline="") as fd, open(trial_out, "w", newline="") as ft:
            wd = csv.DictWriter(fd, fieldnames=DOF_FIELDS)
            wt = csv.DictWriter(ft, fieldnames=TRIAL_FIELDS)
            wd.writeheader(); wt.writeheader()
            for participant, task in trials:
                try:
                    dofs, trial = trial_metrics(
                        ev, runs_root / participant / task / arm,
                        runs_root / participant / task / REFERENCE_TAG,
                        DATASET / "mocap" / "aligned" / participant / task, participant)
                except Exception as exc:
                    failed += 1
                    print(f"  {arm} {participant}/{task}: {type(exc).__name__}: {exc}")
                    continue
                for row in dofs:
                    wd.writerow({"arm": arm, "participant": participant, "task": task, **row})
                wt.writerow({"arm": arm, "participant": participant, "task": task, **trial})
        print(f"{arm}: {len(trials) - failed} trials -> {trial_out}"
              f"{f', {failed} failed' if failed else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
