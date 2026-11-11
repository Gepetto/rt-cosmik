"""Per-trial scores of a run against the reference: causal and aligned error side by side."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
REFERENCE_TAG = "mocap_reference"      # folder name of the reference runs
LIMIT_VIOLATION_RAD = 1e-3

ROW_FIELDS = ["lag", "marker_lag", "sync_r", "causal_rmse_mean", "aligned_rmse_mean",
              "aligned_rmse_median", "causal_bias_abs_mean", "causal_sd_mean", "scored_dof",
              "causal_marker_mm", "aligned_marker_mm", "jitter_deg", "solver_failures",
              "dropped_frames", "violation_pct"]
DOF_FIELDS = ["dof", "causal_rmse_deg", "causal_bias_deg", "causal_sd_deg",
              "aligned_rmse_deg", "aligned_bias_deg", "aligned_sd_deg", "frames"]


def reference_dir(dataset, participant, task):
    """The reference run of a trial: the reference markers through our model and IK."""
    return Path(dataset) / "output" / "campaign" / participant / task / REFERENCE_TAG


def load_eval():
    """``compare_to_mocap.py``, a script rather than a module."""
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint_limits(columns):
    """Lower and upper bounds of the model's configuration at ``columns``."""
    import example_robot_data as robex
    model = robex.human.HumanLoader(height=1.7, weight=70.0, gender="m").robot.model
    return model.lowerPositionLimit[columns], model.upperPositionLimit[columns]


def _by_frame(run_dir):
    """Joint names, angles and markers indexed by video frame (NaN where the run produced none)."""
    run_dir = Path(run_dir)
    with open(run_dir / "joint_angles.csv") as f:
        names = next(csv.reader(f))
    angles = np.genfromtxt(run_dir / "joint_angles.csv", delimiter=",", skip_header=1, ndmin=2)
    with open(run_dir / "markers.csv") as f:
        header = next(csv.reader(f))
    table = np.genfromtxt(run_dir / "markers.csv", delimiter=",", skip_header=1, ndmin=2)
    frames = table[:, header.index("Frame_0")].astype(int)
    span = frames.max() + 1
    q = np.full((span, len(names)), np.nan)
    q[frames] = angles[:len(frames)]
    markers = {}
    for column, name in enumerate(header):
        if name.endswith("_x"):
            markers[name[:-2]] = np.full((span, 3), np.nan)
            markers[name[:-2]][frames] = table[:, column:column + 3]
    return names, q, markers


def _shift(run, reference, lag):
    """Pair run frame ``i + lag`` with reference frame ``i``."""
    if lag >= 0:
        run = run[lag:]
    else:
        reference = reference[-lag:]
    n = min(len(run), len(reference))
    return run[:n], reference[:n]


def _joint_lag(ev, names, q, q_ref):
    """Lag maximising the knee-flexion cross-correlation, and that correlation."""
    columns = [i for i, n in enumerate(names) if ev.SYNC_SIGNAL in n]
    a, b = np.nanmean(q[:, columns], 1), np.nanmean(q_ref[:, columns], 1)
    best, score = 0, -np.inf
    for lag in range(-ev.SYNC_MAX_LAG, ev.SYNC_MAX_LAG + 1):
        x, y = _shift(a, b, lag)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < len(a) // 2:
            continue
        x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        r = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) or np.inf))
        if r > score:
            best, score = lag, r
    return best, score


def _marker_error(markers, ref_markers, lag):
    """Per-frame Euclidean marker errors (m), all markers pooled."""
    errors = []
    for name in set(markers) & set(ref_markers):
        x, y = _shift(markers[name], ref_markers[name], lag)
        e = np.linalg.norm(x - y, axis=1)
        errors.append(e[np.isfinite(e)])
    return np.concatenate(errors) if errors else np.array([])


def _marker_lag(ev, markers, ref_markers):
    """Lag minimising the mean marker error: before IK, so the delay-free signal."""
    span = len(next(iter(markers.values())))
    lags = [lag for lag in range(-ev.SYNC_MAX_LAG, ev.SYNC_MAX_LAG + 1)
            if _marker_error(markers, ref_markers, lag).size > span // 2]
    return min(lags, key=lambda lag: _marker_error(markers, ref_markers, lag).mean())


def _errors(q, q_ref, column, lag):
    x, y = _shift(q[:, column], q_ref[:, column], lag)
    d = (np.degrees(x - y) + 180.0) % 360.0 - 180.0
    return d[np.isfinite(d)]


def _dropped(q, q_ref):
    """Reference frames the run produced nothing for, up to the end of the reference."""
    have = np.zeros(len(q_ref), bool)
    produced = np.isfinite(q[:len(q_ref)]).any(1)
    have[:len(produced)] = produced
    return int((np.isfinite(q_ref).all(1) & ~have).sum())


def _violation_pct(q, columns):
    x = q[:, columns][np.isfinite(q[:, columns]).all(1)]
    lo, hi = joint_limits(columns)
    beyond = (x < lo - LIMIT_VIOLATION_RAD) | (x > hi + LIMIT_VIOLATION_RAD)
    return float(100 * beyond.any(1).mean()) if len(x) else np.nan


def score_trial(run_dir, reference, clock_offset=None):
    """Score one run folder. Returns a ``ROW_FIELDS`` dict and the per-DoF rows.

    Causal error removes only ``clock_offset`` (frames the video trails the
    reference; default ``settings.paper["clock_offset_frames"]``).
    """
    from rtcosmik.config_loader import settings
    if clock_offset is None:
        clock_offset = settings.paper["clock_offset_frames"]
    ev = load_eval()
    run_dir = Path(run_dir)
    names, q, markers = _by_frame(run_dir)
    ref_names, q_ref, ref_markers = _by_frame(reference)
    if names != ref_names:
        raise ValueError(f"{run_dir}: joint columns differ from the reference")
    lag, sync_r = _joint_lag(ev, names, q, q_ref)
    locked = ev.locked_angle_columns()
    scored = [i for i, n in enumerate(names) if n.endswith("[rad]") and n not in locked
              and "quaternion" not in n.lower()]

    dofs = []
    for i in scored:
        row = {"dof": names[i].replace("[rad]", "")}
        for kind, shift in (("causal", clock_offset), ("aligned", lag)):
            d = _errors(q, q_ref, i, shift)
            row.update({f"{kind}_rmse_deg": float(np.sqrt((d ** 2).mean())),
                        f"{kind}_bias_deg": float(d.mean()), f"{kind}_sd_deg": float(d.std())})
        row["frames"] = int(np.isfinite(q[:, i]).sum())
        dofs.append(row)

    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    written = np.isfinite(q).any(1)
    failures = int(info.get("solver_failures", 0)) + int((written & ~np.isfinite(q).all(1)).sum())
    mean = lambda key: round(float(np.mean([d[key] for d in dofs])), 3)
    marker_mm = lambda shift: round(float(_marker_error(markers, ref_markers, shift).mean()) * 1000, 2)
    return {
        "lag": lag - clock_offset,
        "marker_lag": _marker_lag(ev, markers, ref_markers) - clock_offset,
        "sync_r": round(sync_r, 4),
        "causal_rmse_mean": mean("causal_rmse_deg"), "aligned_rmse_mean": mean("aligned_rmse_deg"),
        "aligned_rmse_median": round(float(np.median([d["aligned_rmse_deg"] for d in dofs])), 3),
        "causal_bias_abs_mean": round(float(np.mean([abs(d["causal_bias_deg"]) for d in dofs])), 3),
        "causal_sd_mean": mean("causal_sd_deg"), "scored_dof": len(dofs),
        "causal_marker_mm": marker_mm(clock_offset), "aligned_marker_mm": marker_mm(lag),
        "jitter_deg": round(float(np.nanmedian(np.degrees(np.abs(np.diff(q[:, scored], axis=0))))), 4),
        "solver_failures": failures,
        "dropped_frames": _dropped(q, q_ref),
        "violation_pct": round(_violation_pct(q, scored), 3),
    }, dofs


def score_run(run_dir, dataset, participant, task, clock_offset=None):
    """Score an arm's own run folder against the trial's reference, writing ``per_dof.csv``."""
    row, dofs = score_trial(run_dir, reference_dir(dataset, participant, task), clock_offset)
    with open(Path(run_dir) / "per_dof.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DOF_FIELDS)
        writer.writeheader()
        writer.writerows(dofs)
    return row
