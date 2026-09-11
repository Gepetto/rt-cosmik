"""Meshcat debug visuals for the human model and measured markers.

Optional inspection helpers kept out of the pipeline scripts: joint triads,
model marker positions and one sphere per measured marker.
"""

from typing import Sequence

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin


# -----------------------
# Meshcat debug helpers
# -----------------------


def make_triad_geom(axis_length=0.08, linewidth=2):
    """
    RGB triad as LineSegments:
      X = red, Y = green, Z = blue
    Compatible with meshcat versions that don't have g.Axes.
    """
    # If your meshcat has Axes, use it
    if hasattr(g, "Axes"):
        # Some versions accept axis_radius, some don't. Keep it simple.
        return g.Axes(axis_length=axis_length)

    # Fallback: LineSegments
    # 6 vertices = 3 segments: O->X, O->Y, O->Z
    pts = np.array([
        [0.0, axis_length,  0.0, 0.0,       0.0, 0.0],
        [0.0, 0.0,          0.0, axis_length,0.0, 0.0],
        [0.0, 0.0,          0.0, 0.0,       0.0, axis_length],
    ], dtype=np.float32)

    cols = np.array([
        [255, 255,   0,   0,   0,   0],  # R
        [  0,   0, 255, 255,   0,   0],  # G
        [  0,   0,   0,   0, 255, 255],  # B
    ], dtype=np.uint8)

    geom = g.PointsGeometry(position=pts, color=cols)
    mat  = g.LineBasicMaterial(vertexColors=True, linewidth=linewidth)
    return g.LineSegments(geom, mat)


def make_empty_pointcloud():
    """
    Empty pointcloud node that we can overwrite in update_debug_visuals.
    Works across meshcat versions.
    """
    P = np.zeros((3, 0), dtype=np.float32)
    C = np.zeros((3, 0), dtype=np.uint8)

    if hasattr(g, "PointCloud"):
        return g.PointCloud(P, C)

    # Older versions: render points
    geom = g.PointsGeometry(position=P, color=C)
    mat = g.PointsMaterial(size=0.005, vertexColors=True)
    return g.Points(geom, mat)


def _pin_se3_to_meshcat_tf(M: pin.SE3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T

def setup_debug_visuals(
    vis,
    model: pin.Model,
    marker_names,
    triad_length=0.08,
    triad_radius=0.003,   # gardé pour compat, pas forcément utilisé en fallback
    root="debug",
    clear_root=True,
):
    if clear_root:
        try:
            vis[root].delete()
        except Exception:
            pass

    dbg = {
        "root": root,
        "joint_entries": [],
        "marker_entries": [],
        "model_marker_path": f"{root}/model_markers",
        "missing_marker_frames": [],
    }

    # Create one triad geometry and reuse it
    # linewidth is a best-effort (WebGL may ignore thickness)
    triad = make_triad_geom(axis_length=triad_length, linewidth=max(1, int(triad_radius * 500)))

    # --- joints triads ---
    for jid in range(1, model.njoints):
        jname = model.names[jid]
        path = f"{root}/joints/{jid:04d}_{jname}"  # <= name visible in Meshcat tree
        vis[path].set_object(triad)
        dbg["joint_entries"].append((jid, path))

    # --- marker frame triads ---
    for mk in marker_names:
        try:
            fid = model.getFrameId(mk)
        except Exception:
            fid = None

        if fid is None or fid < 0 or fid >= len(model.frames):
            dbg["missing_marker_frames"].append(mk)
            continue

        path = f"{root}/marker_frames/{fid:04d}_{mk}"  # <= name visible in Meshcat tree
        vis[path].set_object(triad)
        dbg["marker_entries"].append((fid, path))

    # Empty pointcloud node for model markers
    vis[dbg["model_marker_path"]].set_object(make_empty_pointcloud())

    if dbg["missing_marker_frames"]:
        print("[DEBUG] marker frames missing in model (not registered / not added):")
        print("        ", dbg["missing_marker_frames"])

    return dbg


def update_debug_visuals(vis, model: pin.Model, data: pin.Data, q, dbg):
    """
    Update the transforms of all debug triads and refresh the model marker pointcloud.
    Robust to missing keys (won't crash).
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # --- joints ---
    for jid, path in dbg.get("joint_entries", []):
        vis[path].set_transform(_pin_se3_to_meshcat_tf(data.oMi[jid]))

    # --- marker frames ---
    marker_points = []
    for fid, path in dbg.get("marker_entries", []):
        oMf = data.oMf[fid]
        vis[path].set_transform(_pin_se3_to_meshcat_tf(oMf))
        marker_points.append(oMf.translation)

    # --- model marker pointcloud ---
    if marker_points:
        P = np.stack(marker_points, axis=1)  # (3, N)
        C = np.tile(np.array([[0], [255], [0]], dtype=np.uint8), (1, P.shape[1]))
        vis[dbg.get("model_marker_path", "debug/model_markers")].set_object(g.PointCloud(P, C))

# -----------------------
# Named measured markers (debug)
# -----------------------

def setup_measured_markers(vis: "meshcat.Visualizer", marker_names: Sequence[str], radius: float = 0.010, color: int = 0xff0000):
    """Create one small sphere per measured marker, under markers/measured/<name>."""
    sphere = g.Sphere(radius)
    mat = g.MeshPhongMaterial(color=color, opacity=0.9)
    for name in marker_names:
        vis[f"markers/measured/{name}"].set_object(sphere, mat)


def update_measured_markers(vis: "meshcat.Visualizer", mks_dict: dict):
    """Update transforms for the measured marker spheres."""
    for name, p in mks_dict.items():
        try:
            T = tf.translation_matrix(np.asarray(p, dtype=float).reshape(3))
        except Exception:
            continue
        vis[f"markers/measured/{name}"].set_transform(T)
