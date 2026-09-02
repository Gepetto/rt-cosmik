#!/usr/bin/env python3
"""Compare one or more pipeline runs against reference mocap.

Runs are given as ``LABEL=DIRECTORY``; the reference is a mocap trial directory.
Everything is plain CSV in the reference's own column naming, so this works for
any dataset in that format.

    compare_to_mocap.py --reference COMFI/mocap/aligned/1012/Lifting \
        1cam=output/1012/Lifting/1cam_mhe_fatrop \
        2cam=output/1012/Lifting/2cam_mhe_fatrop \
        4cam=output/1012/Lifting/4cam_mhe_fatrop \
        --markers --plots output/1012/eval_Lifting --meshcat

Prints per-joint and per-marker error with a column per run, writes figures with
--plots, and replays every modality together in Meshcat with --meshcat.
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

# Meshcat colours: the reference is white, runs take the rest in order.
REFERENCE_COLOUR = (1.0, 1.0, 1.0)
RUN_COLOURS = [(0.20, 0.60, 0.99), (0.95, 0.45, 0.15), (0.35, 0.80, 0.35),
               (0.85, 0.30, 0.75), (0.95, 0.85, 0.20)]

# Bones drawn between the segment frames of RT-COSMIK's own human model, which
# is the model the joint angles were solved on. Drawing the raw joint chain
# instead collapses the trunk into near-coincident Z/X/Y joint triplets and
# reads as a low, truncated torso. The extremities (toes, fingers, head) have no
# segment frame, but the measured markers are drawn alongside and fill them in.
SKELETON_BONES = [
    ("middle_pelvis", "middle_abdomen"), ("middle_abdomen", "middle_thorax"),
    ("middle_thorax", "middle_head"),
    ("middle_thorax", "left_clavicle"), ("left_clavicle", "left_upperarm"),
    ("left_upperarm", "left_lowerarm"), ("left_lowerarm", "left_hand"),
    ("middle_thorax", "right_clavicle"), ("right_clavicle", "right_upperarm"),
    ("right_upperarm", "right_lowerarm"), ("right_lowerarm", "right_hand"),
    ("middle_pelvis", "left_upperleg"), ("left_upperleg", "left_lowerleg"),
    ("left_lowerleg", "left_foot"),
    ("middle_pelvis", "right_upperleg"), ("right_upperleg", "right_lowerleg"),
    ("right_lowerleg", "right_foot"),
]


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
# Model poses
# ---------------------------------------------------------------------------

class Skeleton:
    """Forward kinematics on RT-COSMIK's human model, for one run.

    Built from the subject the run recorded, then given that run's calibrated
    joint placements so the drawn body matches the model the IK actually solved
    on. Poses are applied as written: the model carries its own root placement,
    so the free-flyer needs no conversion here.
    """

    def __init__(self, info):
        import example_robot_data as robex
        import pinocchio as pin
        self.pin = pin

        subject = info.get("subject") or {}
        self.model = robex.human.HumanLoader(
            height=subject.get("height", 1.80),
            weight=subject.get("weight", 75.0),
            gender=subject.get("gender", "m"),
        ).robot.model

        placements = info.get("joint_placements")
        names = info.get("joint_names")
        if placements and names and len(names) == self.model.njoints:
            for index, (name, translation) in enumerate(zip(names, placements)):
                if name == self.model.names[index]:
                    self.model.jointPlacements[index].translation = np.asarray(
                        translation, dtype=float)

        self.data = self.model.createData()
        self.bones = [(self.model.getFrameId(a), self.model.getFrameId(b))
                      for a, b in SKELETON_BONES
                      if self.model.existFrame(a) and self.model.existFrame(b)] or None

    def segments(self, q):
        """(2N, 3) endpoint pairs for line rendering."""
        self.pin.forwardKinematics(self.model, self.data, q)
        if self.bones is None:  # unknown naming: fall back to the joint chain
            positions = [self.data.oMi[i].translation for i in range(self.model.njoints)]
            pairs = [(self.model.parents[i], i) for i in range(2, self.model.njoints)
                     if self.model.parents[i] > 0]
            return np.array([positions[i] for pair in pairs for i in pair])
        self.pin.updateFramePlacements(self.model, self.data)
        return np.array([self.data.oMf[i].translation
                         for bone in self.bones for i in bone])


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


def freeflyer_to_world(q, root_rotation):
    """Re-express a free-flyer written in a rotated root frame into world axes."""
    if root_rotation is None:
        return q
    R = np.asarray(root_rotation, dtype=float)
    out = np.array(q, dtype=float)
    out[0:3] = R @ q[0:3]
    world = R @ quaternion_matrix(*q[3:7])
    trace = np.trace(world)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (world[2, 1] - world[1, 2]) / s
        qy = (world[0, 2] - world[2, 0]) / s
        qz = (world[1, 0] - world[0, 1]) / s
    else:
        i = int(np.argmax([world[0, 0], world[1, 1], world[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(world[i, i] - world[j, j] - world[k, k] + 1.0) * 2
        parts = [0.0, 0.0, 0.0]
        parts[i] = 0.25 * s
        parts[j] = (world[j, i] + world[i, j]) / s
        parts[k] = (world[k, i] + world[i, k]) / s
        qx, qy, qz = parts
        qw = (world[k, j] - world[j, k]) / s
    out[3:7] = [qx, qy, qz, qw]
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


def compare_joint_angles(run, reference, align_freeflyer=True):
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
    if align_freeflyer and len(translation_cols) == 3:
        # The two models may place their root joint differently (RT-COSMIK's
        # human model carries a fixed base rotation, the reference URDF does
        # not), so the free-flyer is only comparable once expressed in a common
        # frame. Prefer the rotation the run recorded; otherwise fit one.
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

def _print_table(title, row_names, columns, values, unit, width=34):
    print(f"\n{title}")
    header = f"{'':{width}}" + "".join(f"{label:>12}" for label in columns)
    print(header)
    print("-" * len(header))
    for name in row_names:
        line = f"{name:{width}.{width}}"
        for label in columns:
            value = values.get((label, name))
            line += (f"{value:>12.2f}" if value is not None and np.isfinite(value)
                     else f"{'-':>12}")
        print(line)
    print("-" * len(header))
    summary = f"{'median':{width}}"
    for label in columns:
        column = [values[(label, n)] for n in row_names
                  if values.get((label, n)) is not None
                  and np.isfinite(values[(label, n)])]
        summary += f"{np.median(column):>12.2f}" if column else f"{'-':>12}"
    print(summary + f"   {unit}")


def report(results, reference_label):
    labels = [label for label, _ in results]

    joint_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "deg"]
    joint_values = {(label, r["name"]): r["rmse"]
                    for label, data in results for r in data["joints"]}
    _print_table(f"Joint angle RMSE vs {reference_label}", joint_names,
                 labels, joint_values, "deg")

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
    print(f"  {'run':10}{'joint RMSE':>14}{'marker mean':>14}{'cameras':>10}")
    for label, data in results:
        angular = [r["rmse"] for r in data["joints"] if r["unit"] == "deg"]
        markers = [r["mean"] * 1000.0 for r in data["markers"]]
        cameras = data.get("info", {}).get("num_cameras", "-")
        marker_text = f"{np.median(markers):.1f} mm" if markers else "-"
        print(f"  {label:10}{np.median(angular):>10.2f} deg"
              f"{marker_text:>14}{str(cameras):>10}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def write_plots(results, reference, out_dir, reference_label):
    """Write PNG figures summarising the comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in results]
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(labels)))
    height = 0.8 / len(labels)
    written = []

    joint_names = [r["name"] for r in results[0][1]["joints"] if r["unit"] == "deg"]
    figure, axes = plt.subplots(figsize=(13, max(5, 0.24 * len(joint_names))))
    y = np.arange(len(joint_names))
    for i, (label, data) in enumerate(results):
        lookup = {r["name"]: r["rmse"] for r in data["joints"]}
        axes.barh(y + i * height, [lookup.get(n, np.nan) for n in joint_names],
                  height=height, label=label, color=colours[i])
    axes.set_yticks(y + 0.4 - height / 2)
    axes.set_yticklabels([n.replace("[rad]", "").replace("_", " ") for n in joint_names],
                         fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("RMSE vs reference mocap (deg)")
    axes.set_title(f"Joint angle error per degree of freedom  ({reference_label})")
    axes.legend(title="setup")
    axes.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    path = out_dir / "joint_angle_rmse.png"
    figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

    if any(data["markers"] for _, data in results):
        marker_names = [r["name"] for r in results[0][1]["markers"]]
        figure, axes = plt.subplots(figsize=(13, max(5, 0.24 * len(marker_names))))
        y = np.arange(len(marker_names))
        for i, (label, data) in enumerate(results):
            lookup = {r["name"]: r["mean"] * 1000.0 for r in data["markers"]}
            axes.barh(y + i * height, [lookup.get(n, np.nan) for n in marker_names],
                      height=height, label=label, color=colours[i])
        axes.set_yticks(y + 0.4 - height / 2)
        axes.set_yticklabels(marker_names, fontsize=8)
        axes.invert_yaxis()
        axes.set_xlabel("mean 3D position error vs mocap (mm)")
        axes.set_title(f"Marker position error  ({reference_label})")
        axes.legend(title="setup")
        axes.grid(axis="x", alpha=0.3)
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
        axes.boxplot(series, labels=labels, showfliers=False)
        axes.set_ylabel("3D marker error (mm)")
        axes.set_title(f"Distribution of marker error  ({reference_label})")
        axes.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        path = out_dir / "marker_error_distribution.png"
        figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

    interesting = [n for n in joint_names if any(
        k in n for k in ("Knee_Flexion", "Hip_Flexion", "Elbow_Flexion",
                         "Shoulder_Flexion", "Lumbar_Flexion"))][:6]
    if interesting:
        figure, axes_list = plt.subplots(len(interesting), 1,
                                         figsize=(12, 2.1 * len(interesting)), sharex=True)
        axes_list = np.atleast_1d(axes_list)
        ref_index = {n: i for i, n in enumerate(reference["joint_header"])}
        for axes, name in zip(axes_list, interesting):
            axes.plot(np.degrees(reference["joint_values"][:, ref_index[name]]),
                      color="black", lw=1.6, label="mocap", zorder=5)
            for i, (label, data) in enumerate(results):
                index = {n: j for j, n in enumerate(data["joint_header"])}[name]
                axes.plot(np.degrees(data["joint_values"][:, index]),
                          lw=1.0, alpha=0.85, color=colours[i], label=label)
            axes.set_ylabel(name.replace("[rad]", "").replace("_", " ")[:26], fontsize=8)
            axes.grid(alpha=0.3)
        axes_list[0].legend(ncol=len(labels) + 1, fontsize=8)
        axes_list[-1].set_xlabel("frame")
        figure.suptitle(f"Joint angle trajectories vs mocap  ({reference_label})")
        figure.tight_layout()
        path = out_dir / "joint_angle_trajectories.png"
        figure.savefig(path, dpi=130); plt.close(figure); written.append(path)

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


def show_in_meshcat(results, reference, start=0, end=None, step=1,
                    fps=None, manual=True, loop=False, skeletons=True):
    """Replay every modality together in Meshcat.

    Reference markers in white, each run in its own colour, plus a skeleton per
    run posed on that run's own model, so marker error and pose error can be
    judged side by side. The reference contributes its markers: its joint angles
    come from the dataset's own model, which is not ours to pose.
    With `manual`, playback is driven from the terminal.
    """
    import meshcat
    import meshcat.geometry as g

    sources = [("mocap", reference, REFERENCE_COLOUR)]
    for i, (label, data) in enumerate(results):
        sources.append((label, data, RUN_COLOURS[i % len(RUN_COLOURS)]))

    def markers_of(data):
        raw = data.get("markers_raw")
        return raw if isinstance(raw, dict) else (data.get("markers") or {})

    lengths = [len(next(iter(markers_of(d).values())))
               for _, d, _ in sources if markers_of(d)]
    if not lengths:
        print("  Nothing to display: no markers found.")
        return
    end = min(end, min(lengths)) if end else min(lengths)
    fps = fps or (results[0][1].get("info", {}).get("fps") or 40)

    # One skeleton per run, from the model that run recorded.
    skeletons_by_label = {}
    if skeletons:
        for label, data in results:
            try:
                skeletons_by_label[label] = Skeleton(data.get("info") or {})
            except Exception as exc:
                print(f"  Could not build the model for {label}: {exc}")

    vis = meshcat.Visualizer()
    print(f"\nMeshcat: {vis.url()}\n")
    vis["/Background"].set_property("top_color", [0.10, 0.10, 0.12])
    vis["/Background"].set_property("bottom_color", [0.02, 0.02, 0.03])

    print(f"  {'source':10}{'colour (RGB)':24}{'markers':>9}{'skeleton':>10}")
    for label, data, colour in sources:
        has = "yes" if label in skeletons_by_label else "no"
        print(f"  {label:10}{str(tuple(round(c, 2) for c in colour)):24}"
              f"{len(markers_of(data)):>9}{has:>10}")

    def draw(frame):
        for label, data, colour in sources:
            markers = markers_of(data)
            if markers:
                points = np.array([markers[n][frame] for n in sorted(markers)])
                points = points[np.isfinite(points).all(1)]
                if points.size:
                    colours = np.tile(np.array(colour, np.float32).reshape(3, 1),
                                      (1, points.shape[0]))
                    vis[f"trial/{label}/markers"].set_object(
                        g.PointCloud(position=points.T.astype(np.float32),
                                     color=colours, size=0.022))
            skeleton = skeletons_by_label.get(label)
            values = data.get("joint_values")
            if skeleton is not None and values is not None and frame < len(values):
                # Posed on its own model, so q is used exactly as written.
                q = values[frame]
                if len(q) == skeleton.model.nq:
                    points = skeleton.segments(q)
                    if points.size and np.isfinite(points).all():
                        colours = np.tile(np.array(colour, np.float32).reshape(3, 1),
                                          (1, points.shape[0]))
                        geometry = g.PointsGeometry(position=points.T.astype(np.float32),
                                                    color=colours)
                        vis[f"trial/{label}/skeleton"].set_object(g.LineSegments(
                            geometry, g.LineBasicMaterial(vertexColors=True, linewidth=3)))

    if manual and not sys.stdin.isatty():
        print("\n  stdin is not a terminal, playing straight through instead.")
        manual = False

    if not manual:
        try:
            while True:
                for frame in range(start, end, step):
                    tick = time.perf_counter()
                    draw(frame)
                    remaining = step / float(fps) - (time.perf_counter() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
                if not loop:
                    break
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
                    frame, playing = (start, True) if loop else (end - 1, False)
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
                        help="Runs to evaluate; a bare path is labelled by its directory name")
    parser.add_argument("--reference", required=True,
                        help="Reference mocap trial directory (or joint_angles.csv)")
    parser.add_argument("--markers", action="store_true", help="Also compare 3D markers")
    parser.add_argument("--plots", default=None, metavar="DIR",
                        help="Write PNG figures to this directory")
    parser.add_argument("--csv", default=None, help="Write the per-DoF table to this CSV")
    parser.add_argument("--no-align-freeflyer", action="store_true",
                        help="Compare the free-flyer without matching root frames first")

    parser.add_argument("--meshcat", action="store_true",
                        help="Replay every modality together in Meshcat")
    parser.add_argument("--no-skeletons", action="store_true",
                        help="Show only markers in Meshcat, no posed model")
    parser.add_argument("--play", action="store_true",
                        help="Play straight through instead of stepping by hand")
    parser.add_argument("--loop", action="store_true", help="Repeat the replay")
    parser.add_argument("--start", type=int, default=0, help="First frame")
    parser.add_argument("--end", type=int, default=None, help="Last frame")
    parser.add_argument("--step", type=int, default=1, help="Frame stride")
    parser.add_argument("--fps", type=float, default=None, help="Replay rate")
    args = parser.parse_args()

    reference = load_run(args.reference)
    reference_label = Path(args.reference).name

    results = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, Path(label).name
        run = load_run(path)
        joints, freeflyer = compare_joint_angles(
            run, reference, align_freeflyer=not args.no_align_freeflyer)
        results.append((label, {
            "joints": joints, "freeflyer": freeflyer,
            "markers": compare_markers(run, reference) if args.markers else [],
            "info": run["info"], "joint_header": run["joint_header"],
            "joint_values": run["joint_values"], "markers_raw": run["markers"]}))

    report(results, reference_label)

    if args.plots:
        print("\nFigures written:")
        for path in write_plots(results, reference, args.plots, reference_label):
            print(f"  {path}")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "name", "unit", "rmse", "mae"])
            for label, data in results:
                for row in data["joints"]:
                    writer.writerow([label, row["name"], row["unit"], row["rmse"], row["mae"]])
                for row in data["markers"]:
                    writer.writerow([label, row["name"], "m", row["mean"], row["median"]])
        print(f"\nWrote {args.csv}")

    if args.meshcat:
        show_in_meshcat(results, reference, start=args.start,
                        end=args.end, step=args.step, fps=args.fps,
                        manual=not args.play, loop=args.loop,
                        skeletons=not args.no_skeletons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
