#!/usr/bin/env python3
r"""Compare one or more pipeline runs against reference mocap.

Runs are given as LABEL=DIRECTORY; the reference is a mocap trial directory.
Everything is plain CSV in the reference's own column naming, so this works for
any dataset in that format.

    compare_to_mocap.py --reference COMFI/mocap/aligned/1012/Lifting \
        1cam=output/1012/Lifting/1cam_mhe_fatrop \
        2cam=output/1012/Lifting/2cam_mhe_fatrop \
        4cam=output/1012/Lifting/4cam_mhe_fatrop \
        --plots output/1012/eval_Lifting --meshcat

The runs are time-aligned to the mocap first, then compared joint by joint and
marker by marker. --plots writes the figures and the error table, --meshcat
replays every modality as a posed human model. A modality keeps the same colour
and label in every output.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

LINEAR_SUFFIX = "[m]"
ANGULAR_SUFFIX = "[rad]"
FREEFLYER_PREFIX = "Freeflyer"
SYNC_SIGNAL = "Knee_Flexion"          # cross-correlated to time-align the runs
SYNC_MAX_LAG = 120                    # frames searched either way, 3 s at 40 Hz
MODEL_OPACITY = 0.55                  # so overlapping bodies stay readable

# One palette for every output: a modality keeps its colour in the tables, the
# figures and the 3D view. The reference is always the same neutral dark.
REFERENCE_COLOUR = (0.15, 0.15, 0.15)
MODALITY_PALETTE = [
    (0.13, 0.47, 0.85),   # blue
    (0.94, 0.52, 0.13),   # orange
    (0.22, 0.68, 0.34),   # green
    (0.78, 0.24, 0.68),   # magenta
    (0.85, 0.75, 0.15),   # gold
    (0.35, 0.75, 0.80),   # cyan
]


def modality_colour(index):
    return MODALITY_PALETTE[index % len(MODALITY_PALETTE)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _resolve(path, *filenames):
    """Accept a trial directory or a CSV file; try each candidate filename."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in filenames:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"None of {list(filenames)} found in {path}")
    raise FileNotFoundError(f"No such file or directory: {path}")


def _read_table(path):
    """Read a numeric CSV into (header, array), blanks becoming NaN."""
    with open(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(v) if v not in ("", "nan", "NaN", "NA") else math.nan
                 for v in row] for row in reader]
    return header, np.array(rows, dtype=float)


def load_run(path):
    """Load one run or reference: joint angles, markers and any provenance."""
    run = {"path": Path(path)}
    run["joint_header"], run["joint_values"] = _read_table(
        _resolve(path, "joint_angles.csv"))

    try:
        marker_path = _resolve(path, "markers.csv", "markers_trajectories.csv")
        if marker_path.name == "markers_trajectories.csv":
            suffixes, scale = ("_X[mm]", "_Y[mm]", "_Z[mm]"), 0.001
        else:
            suffixes, scale = ("_x", "_y", "_z"), 1.0
        header, values = _read_table(marker_path)
        index = {name: i for i, name in enumerate(header)}
        names = sorted({n[: -len(suffixes[0])] for n in header
                        if n.endswith(suffixes[0])})
        run["markers"] = {n: values[:, [index[n + s] for s in suffixes]] * scale
                          for n in names}
    except FileNotFoundError:
        run["markers"] = {}

    info_path = Path(path) / "run_info.json" if Path(path).is_dir() else None
    run["info"] = json.load(open(info_path)) if info_path and info_path.is_file() else {}
    return run


# ---------------------------------------------------------------------------
# Time synchronisation
# ---------------------------------------------------------------------------

def _sync_signal(header, values):
    """Knee flexion averaged over both legs: a strong, unambiguous gait signal."""
    columns = [i for i, name in enumerate(header) if SYNC_SIGNAL in name]
    if not columns:
        return None
    signal = np.nanmean(values[:, columns], axis=1)
    signal = np.nan_to_num(signal - np.nanmean(signal))
    return signal


def estimate_lag(run, reference, max_lag=SYNC_MAX_LAG):
    """Frames the run trails the reference by, from knee-flexion correlation.

    Returns (lag, correlation). A positive lag means run frame ``i + lag``
    lines up with reference frame ``i``.
    """
    a = _sync_signal(run["joint_header"], run["joint_values"])
    b = _sync_signal(reference["joint_header"], reference["joint_values"])
    if a is None or b is None:
        return 0, float("nan")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        if len(x) < n // 2:
            continue
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator <= 0:
            continue
        score = float(np.dot(x, y) / denominator)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score


def apply_lag(run, reference, lag):
    """Trim a run and the reference so index 0 refers to the same instant."""
    run_start = max(0, lag)
    ref_start = max(0, -lag)
    length = min(len(run["joint_values"]) - run_start,
                 len(reference["joint_values"]) - ref_start)
    if length <= 0:
        return run, reference

    def cut(entry, start):
        out = dict(entry)
        out["joint_values"] = entry["joint_values"][start:start + length]
        out["markers"] = {n: v[start:start + length] for n, v in entry["markers"].items()}
        return out

    return cut(run, run_start), cut(reference, ref_start)


# ---------------------------------------------------------------------------
# Human model
# ---------------------------------------------------------------------------

def build_human_model(info):
    """Rebuild the model a run solved on: RT-COSMIK's human, as calibrated.

    The subject comes from the run, and the calibrated joint placements it
    recorded are applied on top, so the body shown is the one the IK used rather
    than a nominal one built from height and weight alone.
    """
    import example_robot_data as robex

    subject = info.get("subject") or {}
    robot = robex.human.HumanLoader(
        height=subject.get("height", 1.80),
        weight=subject.get("weight", 75.0),
        gender=subject.get("gender", "m"),
    ).robot
    model = robot.model

    placements = info.get("joint_placements")
    names = info.get("joint_names")
    if placements and names and len(names) == model.njoints:
        for index, (name, translation) in enumerate(zip(names, placements)):
            if name == model.names[index]:
                model.jointPlacements[index].translation = np.asarray(
                    translation, dtype=float)

    return model, robot.collision_model, robot.visual_model


def quaternion_matrix(x, y, z, w):
    """Rotation matrix from a quaternion given in (x, y, z, w) order."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not norm:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def freeflyer_to_model_frame(q, root_rotation):
    """Express a free-flyer given in world axes in a model's rotated root frame.

    The reference reports its pelvis pose in world axes, while RT-COSMIK's human
    model places its root joint with a fixed rotation. Rotating the free-flyer
    into that frame lets the same body be posed from either source, so what is
    left on screen is the difference in the joint angles.
    """
    import pinocchio as pin

    rotation = np.asarray(root_rotation, dtype=float)
    out = np.array(q, dtype=float)
    out[0:3] = rotation.T @ np.asarray(q[0:3], dtype=float)
    quaternion = pin.Quaternion(rotation.T @ quaternion_matrix(*q[3:7]))
    out[3:7] = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _trim(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _best_fit_rotation(P, Q):
    """Rotation minimising |R @ P - Q| over corresponding rows (Kabsch)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def locked_angle_columns():
    """Joint-angle column names the configuration has locked, if any.

    A locked DoF is held at zero because the marker set cannot observe it, so
    scoring it against mocap measures the lock, not the pipeline, and drags the
    summary toward whatever the subject happened to do with that joint. The
    columns are still listed, marked, so the exclusion is visible rather than
    silent.

    Returns an empty set for an ordinary run, which leaves every number here
    exactly as it was before locking existed.
    """
    try:
        from rtcosmik.config_loader import settings
        import example_robot_data as robex
    except Exception:
        return set()
    locked = list(getattr(settings, "locked_joints", ()) or ())
    if not locked:
        return set()
    names = list(settings.joint_angles_names)
    model = robex.human.HumanLoader(
        height=settings.human_height, weight=settings.human_weight,
        gender=settings.human_gender).robot.model
    columns = set()
    for joint in locked:
        if not model.existJointName(joint):
            continue
        first = model.joints[model.getJointId(joint)].idx_q
        span = model.joints[model.getJointId(joint)].nq
        # joint_angles_names is written in q order, so idx_q indexes it directly.
        columns.update(names[first:first + span])
    return columns


def compare_joint_angles(run, reference):
    """Per-DoF error. Returns (rows, freeflyer_info)."""
    header, est = run["joint_header"], run["joint_values"]
    ref_header, ref = reference["joint_header"], reference["joint_values"]
    if header != ref_header:
        raise ValueError(
            "Joint angle columns differ.\n"
            f"  only in run      : {[c for c in header if c not in ref_header]}\n"
            f"  only in reference: {[c for c in ref_header if c not in header]}")
    est, ref = _trim(est, ref)

    translation_cols = [i for i, n in enumerate(header)
                        if n.startswith(FREEFLYER_PREFIX) and n.endswith(LINEAR_SUFFIX)]
    quaternion_cols = [i for i, n in enumerate(header) if "quaternion" in n.lower()]

    freeflyer = {}
    est = est.copy()
    if len(translation_cols) == 3:
        # The two models may place their root joint differently (RT-COSMIK's
        # human model carries a fixed base rotation, the reference does not), so
        # the free-flyer is only comparable once expressed in a common frame.
        recorded = run.get("info", {}).get("root_placement_rotation")
        P, Q = est[:, translation_cols], ref[:, translation_cols]
        valid = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        if recorded is not None:
            R, source = np.asarray(recorded, dtype=float), "model root placement"
        elif valid.sum() >= 3:
            R = _best_fit_rotation(P[valid] - P[valid].mean(0), Q[valid] - Q[valid].mean(0))
            source = "fitted to the data"
        else:
            R, source = np.eye(3), "none"
        est[:, translation_cols] = (R @ P.T).T
        freeflyer = {
            "rotation_deg": math.degrees(math.acos(
                max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))),
            "source": source,
        }
        if len(quaternion_cols) == 4:
            angles = []
            for qe, qr in zip(est[:, quaternion_cols], ref[:, quaternion_cols]):
                if np.isfinite(qe).all() and np.isfinite(qr).all():
                    relative = (R @ quaternion_matrix(*qe)).T @ quaternion_matrix(*qr)
                    angles.append(math.degrees(math.acos(
                        max(-1.0, min(1.0, (np.trace(relative) - 1) / 2)))))
            freeflyer["orientation_deg"] = float(np.mean(angles)) if angles else math.nan

    locked = locked_angle_columns()
    rows = []
    for index, name in enumerate(header):
        if index in quaternion_cols:
            continue
        difference = est[:, index] - ref[:, index]
        if name.endswith(ANGULAR_SUFFIX):
            # Wrap to (-180, 180] so a rollover is not counted as a huge error.
            difference = (np.degrees(difference) + 180.0) % 360.0 - 180.0
            unit = "deg"
        elif name.endswith(LINEAR_SUFFIX):
            unit = "m"
        else:
            unit = ""
        difference = difference[np.isfinite(difference)]
        if difference.size:
            rows.append({"name": name, "unit": unit,
                         "locked": name in locked,
                         "rmse": float(np.sqrt((difference ** 2).mean())),
                         "mae": float(np.abs(difference).mean())})
    return rows, freeflyer


def compare_markers(run, reference):
    """Per-marker Euclidean error in metres."""
    rows = []
    for name in sorted(set(run["markers"]) & set(reference["markers"])):
        est, ref = _trim(run["markers"][name], reference["markers"][name])
        valid = np.isfinite(est).all(1) & np.isfinite(ref).all(1)
        if not valid.any():
            continue
        error = np.linalg.norm(est[valid] - ref[valid], axis=1)
        rows.append({"name": name, "mean": float(error.mean()),
                     "median": float(np.median(error)),
                     "p95": float(np.percentile(error, 95))})
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(title, row_names, columns, values, unit, width=34, excluded=()):
    """Print one comparison table.

    ``excluded`` rows are still printed, marked with a trailing dot, but left out
    of the mean and median: they are DoF the configuration locked, so their error
    describes the lock rather than the pipeline. Showing them keeps the exclusion
    auditable instead of hiding rows the reader might expect to find.
    """
    excluded = set(excluded)
    scored = [n for n in row_names if n not in excluded]
    print(f"\n{title}")
    header = f"{'':{width}}" + "".join(f"{label:>12}" for label in columns)
    print(header)
    print("-" * len(header))
    for name in row_names:
        marker = " ." if name in excluded else ""
        line = f"{name:{width - len(marker)}.{width - len(marker)}}{marker}"
        for label in columns:
            value = values.get((label, name))
            line += (f"{value:>12.2f}" if value is not None and np.isfinite(value)
                     else f"{'-':>12}")
        print(line)
    print("-" * len(header))
    for stat, function in (("mean", np.mean), ("median", np.median)):
        line = f"{stat:{width}}"
        for label in columns:
            column = [values[(label, n)] for n in scored
                      if values.get((label, n)) is not None
                      and np.isfinite(values[(label, n)])]
            line += f"{function(column):>12.2f}" if column else f"{'-':>12}"
        print(line + (f"   {unit}" if stat == "median" else ""))
    if excluded & set(row_names):
        print(f"  . locked DoF, excluded from the mean and median "
              f"({len(scored)} of {len(row_names)} scored)")


def report(results, reference_label):
    labels = [label for label, _ in results]

    joint_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "deg"]
    joint_values = {(label, r["name"]): r["rmse"]
                    for label, data in results for r in data["joints"]}
    locked = {r["name"] for _, data in results for r in data["joints"]
              if r.get("locked")}
    _print_table(f"Joint angle RMSE vs {reference_label}", joint_names,
                 labels, joint_values, "deg", excluded=locked)

    linear_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "m"]
    if linear_names:
        linear_values = {(label, r["name"]): r["rmse"] * 1000.0
                         for label, data in results for r in data["joints"]}
        _print_table("Free-flyer translation RMSE (aligned to the reference frame)",
                     linear_names, labels, linear_values, "mm")
        print()
        for label, data in results:
            ff = data.get("freeflyer") or {}
            if ff:
                print(f"  {label:8} root frame rotation removed: "
                      f"{ff['rotation_deg']:6.2f} deg ({ff['source']});  "
                      f"root orientation error: "
                      f"{ff.get('orientation_deg', float('nan')):.2f} deg")

    if any(data["markers"] for _, data in results):
        marker_names = [r["name"] for r in results[0][1]["markers"]]
        marker_values = {(label, r["name"]): r["mean"] * 1000.0
                         for label, data in results for r in data["markers"]}
        _print_table("Marker position error, mean per marker", marker_names,
                     labels, marker_values, "mm")

    print("\nSummary")
    print(f"  {'run':10}{'joints mean':>13}{'markers mean':>15}"
          f"{'lag':>8}{'cameras':>9}")
    for label, data in results:
        angular = [r["rmse"] for r in data["joints"] if r["unit"] == "deg"]
        markers = [r["mean"] * 1000.0 for r in data["markers"]]
        cameras = data.get("info", {}).get("num_cameras", "-")
        marker_text = f"{np.mean(markers):.1f} mm" if markers else "-"
        lag = data.get("lag")
        lag_text = f"{lag:+d}" if lag is not None else "-"
        print(f"  {label:10}{np.mean(angular):>9.2f} deg{marker_text:>15}"
              f"{lag_text:>8}{str(cameras):>9}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _bar_chart(axes, names, results, lookup, colours, xlabel, title,
               mean_names=None):
    """Grouped horizontal bars, with an overall mean at the top.

    `mean_names` restricts what the mean averages, so a summary row such as the
    free-flyer can be charted without being folded into it.
    """
    mean_names = list(names if mean_names is None else mean_names)
    rows = ["MEAN (all)"] + list(names)
    y = np.arange(len(rows))
    height = 0.8 / len(results)
    for i, (label, data) in enumerate(results):
        table = lookup(data)
        values = [table.get(n, np.nan) for n in names]
        finite = [table.get(n, np.nan) for n in mean_names]
        finite = [v for v in finite if np.isfinite(v)]
        axes.barh(y + i * height, [np.mean(finite) if finite else np.nan] + values,
                  height=height, label=label, color=colours[i])
    axes.set_yticks(y + 0.4 - height / 2)
    axes.set_yticklabels(rows, fontsize=8)
    # Set the aggregate apart from the per-item rows it summarises.
    axes.get_yticklabels()[0].set_fontweight("bold")
    axes.axhline(0.8, color="0.4", lw=1.0, ls="--")
    axes.invert_yaxis()
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    axes.legend(title="modality")
    axes.grid(axis="x", alpha=0.3)


def write_plots(results, reference, out_dir, reference_label, colours):
    """Write PNG figures summarising the comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    joint_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "deg"]
    short = [n.replace("[rad]", "").replace("_", " ") for n in joint_names]
    ORIENTATION = "Freeflyer orientation"

    def joint_lookup(data):
        table = {r["name"]: r["rmse"] for r in data["joints"]}
        table[ORIENTATION] = (data.get("freeflyer") or {}).get(
            "orientation_deg", np.nan)
        return table

    rows = joint_names + [ORIENTATION]
    figure, axes = plt.subplots(figsize=(13, max(6, 0.26 * (len(rows) + 2))))
    _bar_chart(axes, rows, results, joint_lookup, colours,
               "RMSE vs reference mocap (deg)",
               f"Joint angle error per degree of freedom  ({reference_label})",
               mean_names=joint_names)
    axes.set_yticklabels(["MEAN (all joints)"] + short + [ORIENTATION], fontsize=8)
    axes.get_yticklabels()[0].set_fontweight("bold")
    figure.tight_layout()
    path = out_dir / "joint_angle_rmse.png"
    figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

    if any(data["markers"] for _, data in results):
        marker_names = [r["name"] for r in results[0][1]["markers"]]
        TRANSLATION = "Freeflyer translation"

        def marker_lookup(data):
            table = {r["name"]: r["mean"] * 1000.0 for r in data["markers"]}
            linear = [r["rmse"] * 1000.0 for r in data["joints"] if r["unit"] == "m"]
            table[TRANSLATION] = float(np.mean(linear)) if linear else np.nan
            return table

        rows = marker_names + [TRANSLATION]
        figure, axes = plt.subplots(figsize=(13, max(6, 0.26 * (len(rows) + 2))))
        _bar_chart(axes, rows, results, marker_lookup, colours,
                   "error vs mocap (mm)",
                   f"Marker position error  ({reference_label})",
                   mean_names=marker_names)
        axes.set_yticklabels(["MEAN (all markers)"] + marker_names + [TRANSLATION],
                             fontsize=8)
        axes.get_yticklabels()[0].set_fontweight("bold")
        figure.tight_layout()
        path = out_dir / "marker_error.png"
        figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

        figure, axes = plt.subplots(figsize=(9, 5))
        series = []
        for label, data in results:
            errors = []
            for name in [r["name"] for r in data["markers"]]:
                est, ref = _trim(data["markers_raw"][name], reference["markers"][name])
                valid = np.isfinite(est).all(1) & np.isfinite(ref).all(1)
                errors.append(np.linalg.norm(est[valid] - ref[valid], axis=1) * 1000.0)
            series.append(np.concatenate(errors) if errors else np.array([]))
        boxes = axes.boxplot(series, labels=[l for l, _ in results],
                             showfliers=False, patch_artist=True)
        for patch, colour in zip(boxes["boxes"], colours):
            patch.set_facecolor(colour); patch.set_alpha(0.65)
        axes.set_ylabel("3D marker error (mm)")
        axes.set_title(f"Distribution of marker error  ({reference_label})")
        axes.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        path = out_dir / "marker_error_distribution.png"
        figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

    # The free-flyer on its own, since its translation and its orientation
    # cannot share an axis with each other or with the joint angles.
    linear_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "m"]
    if linear_names:
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})

        short_linear = [n.replace("Freeflyer_", "").replace("[m]", "")
                        for n in linear_names]
        y = np.arange(len(linear_names) + 1)
        height = 0.8 / len(results)
        for i, (label, data) in enumerate(results):
            table = {r["name"]: r["rmse"] * 1000.0 for r in data["joints"]}
            values = [table.get(n, np.nan) for n in linear_names]
            finite = [v for v in values if np.isfinite(v)]
            left.barh(y + i * height, [np.mean(finite) if finite else np.nan] + values,
                      height=height, label=label, color=colours[i])
        left.set_yticks(y + 0.4 - height / 2)
        left.set_yticklabels(["MEAN"] + short_linear, fontsize=9)
        left.get_yticklabels()[0].set_fontweight("bold")
        left.axhline(0.8, color="0.4", lw=1.0, ls="--")
        left.invert_yaxis()
        left.set_xlabel("translation RMSE vs mocap (mm)")
        left.set_title("Free-flyer translation")
        left.legend(title="modality", fontsize=8)
        left.grid(axis="x", alpha=0.3)

        for i, (label, data) in enumerate(results):
            angle = (data.get("freeflyer") or {}).get("orientation_deg", np.nan)
            right.barh([i * height], [angle], height=height,
                       label=label, color=colours[i])
        right.set_yticks([0.4 - height / 2])
        right.set_yticklabels(["orientation"], fontsize=9)
        right.invert_yaxis()
        right.set_xlabel("orientation error vs mocap (deg)")
        right.set_title("Free-flyer orientation")
        right.grid(axis="x", alpha=0.3)

        figure.suptitle(f"Free-flyer error  ({reference_label})")
        figure.tight_layout()
        path = out_dir / "freeflyer_error.png"
        figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

    # Every joint angle over time, each panel captioned with its own error.
    columns = 3
    rows = int(math.ceil(len(joint_names) / columns))
    figure, grid = plt.subplots(rows, columns, figsize=(6.2 * columns, 2.3 * rows),
                                squeeze=False)
    ref_index = {n: i for i, n in enumerate(reference["joint_header"])}
    for position, name in enumerate(joint_names):
        axes = grid[position // columns][position % columns]
        axes.plot(np.degrees(reference["joint_values"][:, ref_index[name]]),
                  color=REFERENCE_COLOUR, lw=1.5, label="mocap", zorder=5)
        captions = []
        for i, (label, data) in enumerate(results):
            index = {n: j for j, n in enumerate(data["joint_header"])}[name]
            axes.plot(np.degrees(data["joint_values"][:, index]),
                      lw=1.0, alpha=0.85, color=colours[i], label=label)
            rmse = {r["name"]: r["rmse"] for r in data["joints"]}.get(name, float("nan"))
            captions.append(f"{label} {rmse:.1f}")
        axes.set_title(f"{name.replace('[rad]', '').replace('_', ' ')}\n"
                       f"RMSE  " + "   ".join(captions) + "  deg", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.grid(alpha=0.3)
    for empty in range(len(joint_names), rows * columns):
        grid[empty // columns][empty % columns].axis("off")
    grid[0][0].legend(fontsize=7, ncol=len(results) + 1)
    figure.suptitle(f"Joint angle trajectories vs mocap  ({reference_label})",
                    fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    path = out_dir / "joint_angle_trajectories.png"
    figure.savefig(path, dpi=110); plt.close(figure); written.append(path)

    return written


# ---------------------------------------------------------------------------
# Meshcat replay
# ---------------------------------------------------------------------------

class KeyReader:
    """Read single keypresses without waiting for Enter, when on a terminal."""

    def __enter__(self):
        self._fd = None
        try:
            import termios, tty
            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._fd = None
        return self

    def get(self):
        if self._fd is None:
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        if self._fd is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)


def show_in_meshcat(results, reference, colours):
    """Replay every modality together in Meshcat.

    Every source is drawn as a posed human model, tinted with the colour it has
    everywhere else, the reference in near-black. All of them use the same body,
    so what differs on screen is the joint angles, not the anthropometry.
    With `manual`, playback is driven from the terminal.
    """
    import meshcat
    import meshcat.geometry as g
    from pinocchio.visualize import MeshcatVisualizer

    sources = [("mocap", reference, REFERENCE_COLOUR)]
    for (label, data), colour in zip(results, colours):
        sources.append((label, data, colour))

    lengths = [len(d["markers"][next(iter(d["markers"]))])
               for _, d, _ in sources if d.get("markers")]
    if not lengths:
        print("  Nothing to display: no markers found.")
        return
    start, step, end = 0, 1, min(lengths)
    fps = results[0][1].get("info", {}).get("fps") or 40

    viewer = meshcat.Visualizer()
    print(f"\nMeshcat: {viewer.url()}\n")
    # Light background so the reference reads dark here as it does in the plots.
    viewer["/Background"].set_property("top_color", [0.96, 0.96, 0.97])
    viewer["/Background"].set_property("bottom_color", [0.80, 0.82, 0.85])

    # The reference is posed on the same body as the runs, since its joint
    # angles are the same 43 degrees of freedom in the same order. Only its
    # free-flyer needs rotating into the model's root frame.
    reference_info = results[0][1].get("info") or {}
    visualizers = {}
    for label, data, colour in sources:
        info = reference_info if label == "mocap" else (data.get("info") or {})
        try:
            model, collision_model, visual_model = build_human_model(info)
            visualizer = MeshcatVisualizer(model, collision_model, visual_model)
            visualizer.initViewer(viewer)
            visualizer.loadViewerModel(rootNodeName=f"model_{label}",
                                       color=list(colour) + [MODEL_OPACITY])
            root = (np.asarray(model.jointPlacements[1].rotation, dtype=float)
                    if label == "mocap" else None)
            visualizers[label] = (visualizer, model, root)
        except Exception as exc:
            print(f"  Could not build the model for {label}: {exc}")

    print(f"  {'modality':10}{'colour (RGB)':24}{'markers':>9}{'model':>8}{'lag':>7}")
    for label, data, colour in sources:
        lag = data.get("lag")
        print(f"  {label:10}{str(tuple(round(c, 2) for c in colour)):24}"
              f"{len(data.get('markers') or {}):>9}"
              f"{('yes' if label in visualizers else 'no'):>8}"
              f"{(f'{lag:+d}' if lag is not None else '-'):>7}")

    def draw(frame):
        for label, data, colour in sources:
            markers = data.get("markers") or {}
            if markers:
                points = np.array([markers[n][frame] for n in sorted(markers)])
                points = points[np.isfinite(points).all(1)]
                if points.size:
                    tint = np.tile(np.array(colour, np.float32).reshape(3, 1),
                                   (1, points.shape[0]))
                    viewer[f"markers/{label}"].set_object(
                        g.PointCloud(position=points.T.astype(np.float32),
                                     color=tint, size=0.022))
            entry = visualizers.get(label)
            values = data.get("joint_values")
            if entry is not None and values is not None and frame < len(values):
                visualizer, model, to_root_frame = entry
                q = np.asarray(values[frame], dtype=float)
                if to_root_frame is not None:
                    q = freeflyer_to_model_frame(q, to_root_frame)
                if len(q) == model.nq:
                    visualizer.display(q)

    if not sys.stdin.isatty():
        # No terminal to take keys from, so just play it through once.
        print("\n  stdin is not a terminal, playing straight through instead.")
        try:
            for frame in range(start, end, step):
                tick = time.perf_counter()
                draw(frame)
                remaining = step / float(fps) - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n  stopped")
        return

    print("\n  space play/pause   n next   p previous   f/b jump 25   "
          "[/] speed   r restart   q quit")
    frame, playing, speed = start, False, 1.0
    draw(frame)

    def status():
        print(f"\r  frame {frame:6d}/{end - 1}  t={frame / fps:6.2f}s  "
              f"{'playing' if playing else 'paused '}  x{speed:g}   ",
              end="", flush=True)

    status()
    with KeyReader() as keys:
        while True:
            key = keys.get()
            if key:
                if key == "q":
                    break
                if key == " ":
                    playing = not playing
                elif key == "n":
                    playing, frame = False, min(end - 1, frame + step)
                elif key == "p":
                    playing, frame = False, max(start, frame - step)
                elif key == "f":
                    playing, frame = False, min(end - 1, frame + 25)
                elif key == "b":
                    playing, frame = False, max(start, frame - 25)
                elif key == "]":
                    speed = min(8.0, speed * 1.5)
                elif key == "[":
                    speed = max(0.125, speed / 1.5)
                elif key == "r":
                    frame = start
                draw(frame)
                status()
            if playing:
                frame += step
                if frame >= end:
                    frame, playing = start, True
                draw(frame)
                status()
                time.sleep(step / (float(fps) * speed))
            else:
                time.sleep(0.02)
    print("\n  stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="LABEL=DIR",
                        help="Runs to compare; a bare path is labelled by its directory name")
    parser.add_argument("--reference", required=True,
                        help="Reference mocap trial directory")
    parser.add_argument("--plots", default=None, metavar="DIR",
                        help="Write the figures and the error table to this directory")
    parser.add_argument("--meshcat", action="store_true",
                        help="Replay every modality as a posed human model in Meshcat")
    args = parser.parse_args()

    reference = load_run(args.reference)
    reference_label = Path(args.reference).name

    loaded = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, Path(label).name
        loaded.append((label, load_run(path)))

    # The cameras are synchronised with each other but not with the mocap, so
    # one common offset covers them all. Estimate it per run from knee flexion
    # and use the median, which is robust to a run whose correlation is poor.
    print("Time alignment (knee flexion correlation)")
    estimates = []
    for label, run in loaded:
        run_lag, score = estimate_lag(run, reference)
        estimates.append(run_lag)
        print(f"  {label:10}{run_lag:+5d} frames   correlation {score:.3f}")
    lag = int(np.median(estimates)) if estimates else 0
    print(f"  applying a common lag of {lag:+d} frames to every modality")

    results = []
    for label, run in loaded:
        aligned_run, aligned_reference = apply_lag(run, reference, lag)
        joints, freeflyer = compare_joint_angles(aligned_run, aligned_reference)
        results.append((label, {
            "joints": joints, "freeflyer": freeflyer,
            "markers": compare_markers(aligned_run, aligned_reference),
            "info": run["info"], "joint_header": aligned_run["joint_header"],
            "joint_values": aligned_run["joint_values"],
            "markers_raw": aligned_run["markers"], "lag": lag}))
    _, reference = apply_lag(loaded[0][1], reference, lag)

    colours = [modality_colour(i) for i in range(len(results))]
    report(results, reference_label)

    if args.plots:
        print("\nWritten:")
        for path in write_plots(results, reference, args.plots, reference_label, colours):
            print(f"  {path}")
        table = Path(args.plots) / "errors.csv"
        with open(table, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "name", "unit", "rmse", "mae"])
            for label, data in results:
                for row in data["joints"]:
                    writer.writerow([label, row["name"], row["unit"], row["rmse"], row["mae"]])
                for row in data["markers"]:
                    writer.writerow([label, row["name"], "m", row["mean"], row["median"]])
        print(f"  {table}")

    if args.meshcat:
        display = [(label, {**data, "markers": data["markers_raw"]})
                   for label, data in results]
        show_in_meshcat(display, reference, colours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
