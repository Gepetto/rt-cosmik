"""Benchmark / compare the MHE inverse-kinematics backends on recorded markers.

The *same* sliding-window MHE IK problem is solved with both backends and we
report, for each, marker-tracking RMSE and per-frame solve time, plus the
agreement between the two solutions (the equivalence check):

  * ``fatrop`` -> rtcosmik.ik.ik.RT_SWIKA_FATROP         (validated reference)
  * ``acados`` -> rtcosmik.ik.ik.RT_SWIKA_ACADOS  (new backend)

The human model is built exactly like the live pipeline (HumanLoader -> scale ->
register -> IPOPT init pose -> recalibrate), so the comparison reflects what is
deployed. With ``--display`` it shows the human model and measured/model markers
in meshcat, as usually done in RT-COSMIK.

acados is auto-skipped if ``acados_template`` / ``ACADOS_SOURCE_DIR`` are not
available, so the script also runs fatrop-only.

Example
-------
    python3 tests/benchmark/benchmark_mhe_ik_backends.py --display
"""

import os
import sys
import time
import math
import argparse
from collections import deque

import numpy as np
import pandas as pd
import pinocchio as pin
import example_robot_data as robex

# Adjust Python path to find settings.py (repo root) and src.rtcosmik.*
cosmik_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, cosmik_path)
sys.path.insert(0, os.path.join(cosmik_path, "src"))

from settings import Settings
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.ik import ik as ik_module
from rtcosmik.human_model.model_utils import (
    scale_human_model,
    mks_registration,
    recalibrate_marker_frames_in_joint_space,
)


# ---------------------------------------------------------------------------
# Simple running stats helper (same style as benchmark_full_pipeline_all.py)
# ---------------------------------------------------------------------------

class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.max = float("-inf")
        self._values = []

    def add(self, x: float):
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        self.max = max(self.max, x)
        self._values.append(x)

    @property
    def mean(self):
        return self.sum / self.n if self.n > 0 else float("nan")

    @property
    def std(self):
        if self.n <= 1:
            return float("nan")
        m = self.mean
        return math.sqrt(max(self.sumsq / self.n - m * m, 0.0))

    @property
    def median(self):
        return float(np.median(self._values)) if self._values else float("nan")


# ---------------------------------------------------------------------------
# Marker name map: COSMIK name -> name in the CSV.
#
# The model build (mks_registration) needs the FULL COSMIK marker set, so this
# map must cover every COSMIK marker. Core body markers map 1:1; entries marked
# "approx" have no exact CSV equivalent (different protocol) and only set
# marker-frame placements for thorax/hands/head. As those are not in
# KEYS_TO_TRACK by default, the approximations do not affect the comparison.
# ---------------------------------------------------------------------------

MARKER_NAME_MAP = {
    # pelvis
    "RASI": "RASI", "LASI": "LASI", "RPSI": "RPSI", "LPSI": "LPSI",
    "T11": "TV12",                                              # approx (T11 ~ TV12)
    # thorax / torso
    "C7": "C7", "T6": "TV8",                                    # approx (T6 ~ TV8)
    "RSHO": "RSHO", "LSHO": "LSHO",
    # arms
    "RELB": "RELB", "RMELB": "RMELB", "RWRI": "RWRI", "RMWRI": "RMWRI",
    "LELB": "LELB", "LMELB": "LMELB", "LWRI": "LWRI", "LMWRI": "LMWRI",
    # hands (approx: CSV hand cluster RHM2/RHM5/RHAND)
    "RTHU": "RHM2", "RMID": "RHAND", "RPIN": "RHM5",
    "LTHU": "LHM2", "LMID": "LHAND", "LPIN": "LHM5",
    # legs
    "RKNE": "RKNE", "RMKNE": "RMKNE", "RANK": "RANK", "RMANK": "RMANK",
    "LKNE": "LKNE", "LMKNE": "LMKNE", "LANK": "LANK", "LMANK": "LMANK",
    # feet
    "RTOE": "RTOE", "R5MHD": "R5MHD", "RHEE": "RHEE",
    "LTOE": "LTOE", "L5MHD": "L5MHD", "LHEE": "LHEE",
    # head (approx: CSV 4-marker head cluster RHD/LHD/FHD/BHD; no eye markers)
    "Head": "BHD", "Nose": "FHD", "REar": "RHD", "LEar": "LHD",
    "REye": "FHD", "LEye": "FHD",
}

# Markers actually tracked by the IK (must be registered frames). Reliable body
# subset, excluding the approximate head/hand markers.
KEYS_TO_TRACK = [
    "RASI", "LASI", "RPSI", "LPSI", "C7", "RSHO", "LSHO",
    "RELB", "RMELB", "RWRI", "RMWRI", "LELB", "LMELB", "LWRI", "LMWRI",
    "RKNE", "RMKNE", "RANK", "RMANK", "RTOE", "R5MHD", "RHEE",
    "LKNE", "LMKNE", "LANK", "LMANK", "LTOE", "L5MHD", "LHEE",
]


# ---------------------------------------------------------------------------
# CSV loading (wide format, millimetres)
# ---------------------------------------------------------------------------

def load_markers(csv_path):
    """Return a list of per-frame dicts {COSMIK_name: xyz[m]} via MARKER_NAME_MAP."""
    df = pd.read_csv(csv_path)
    cols = {csv: tuple(f"{csv}_{ax}[mm]" for ax in "XYZ")
            for csv in set(MARKER_NAME_MAP.values())}
    missing = [c for csv in cols for c in cols[csv] if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns required by MARKER_NAME_MAP: {missing}")

    frames = []
    for _, row in df.iterrows():
        frames.append({
            cos: np.array([row[cols[csv][0]], row[cols[csv][1]], row[cols[csv][2]]],
                          dtype=float) / 1000.0
            for cos, csv in MARKER_NAME_MAP.items()
        })
    return frames


# ---------------------------------------------------------------------------
# Model build (identical to the live pipeline)
# ---------------------------------------------------------------------------

def build_model(mks0, settings, keys):
    robot = robex.human.HumanLoader(
        height=settings.human_height, weight=settings.human_weight,
        gender=settings.human_gender).robot
    model = robot.model
    model = scale_human_model(model, mks0, gender=settings.human_gender,
                              subject_height=settings.human_height)
    model = mks_registration(model, mks0, gender=settings.human_gender,
                             subject_height=settings.human_height)

    keys = [k for k in keys if model.existFrame(k)]
    if not keys:
        raise RuntimeError("None of KEYS_TO_TRACK are registered frames in the model.")

    # initial pose via a single IPOPT solve (sample-by-sample IK)
    omega = {k: 1.0 for k in keys}
    q0 = RT_IK(model, mks0, pin.neutral(model), keys, settings.dt, omega
               ).solve_ik_sample_casadi()
    q0 = np.asarray(q0).flatten()

    # recalibrate marker frames so offsets match the measured markers at q0
    model = recalibrate_marker_frames_in_joint_space(
        model, q0, mks0, list(MARKER_NAME_MAP.keys()))
    return robot, model, keys, q0


# ---------------------------------------------------------------------------
# meshcat visualization (human model + measured/model markers)
# ---------------------------------------------------------------------------

def make_visualizer(robot, model, keys):
    """Custom meshcat viz that does not need hpp-fcl: loads the human STL meshes
    directly and shows the measured (red) and model (black) markers."""
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf

    vis = meshcat.Visualizer()
    print(f"[viz] open meshcat at: {vis.url()}")
    vis["/Background"].set_property("top_color", [1, 1, 1])
    vis["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

    geoms = []  # (name, parent_joint_id, jMg_homogeneous, meshScale)
    for go in robot.visual_model.geometryObjects:
        try:
            mesh = g.StlMeshGeometry.from_file(go.meshPath)
        except Exception:
            continue
        vis[f"human/{go.name}"].set_object(
            mesh, g.MeshLambertMaterial(color=0xBBBBBB, opacity=0.5))
        geoms.append((go.name, go.parentJoint,
                      np.array(go.placement.homogeneous),
                      np.asarray(go.meshScale).flatten()))

    for key in keys:
        vis[f"meas/{key}"].set_object(
            g.Sphere(0.02), g.MeshLambertMaterial(color=0xFF0000, opacity=0.7))
        vis[f"model/{key}"].set_object(
            g.Sphere(0.015), g.MeshLambertMaterial(color=0x111111, opacity=1.0))
    return {"vis": vis, "tf": tf, "geoms": geoms}


def show_frame(viz, model, data, q, mks, keys):
    """Update mesh + marker transforms (data must already hold FK for q)."""
    vis, tf, geoms = viz["vis"], viz["tf"], viz["geoms"]
    for name, jid, jMg, scale in geoms:
        T = data.oMi[jid].homogeneous @ jMg @ np.diag([scale[0], scale[1], scale[2], 1.0])
        vis[f"human/{name}"].set_transform(T)
    for key in keys:
        vis[f"meas/{key}"].set_transform(tf.translation_matrix(mks[key]))
        p = np.asarray(data.oMf[model.getFrameId(key)].translation)
        vis[f"model/{key}"].set_transform(tf.translation_matrix(p))


# ---------------------------------------------------------------------------
# Run one backend over the whole sequence (pipeline sliding-window logic)
# ---------------------------------------------------------------------------

def make_solver(backend, model, keys, settings, export_dir):
    if backend == "fatrop":
        return RT_SWIKA_FATROP(model, keys, settings.N, code="python")
    return RT_SWIKA_ACADOS(model, keys, settings.N, settings.dt,
                           export_dir=export_dir, acados_source_dir=None)


def run_backend(backend, model, keys, frames, q0, settings, export_dir,
                viz=None):
    nq, nv = model.nq, model.nv
    cost_weights = np.asarray(settings.cost_weights, dtype=float)
    N, dt = settings.N, settings.dt

    print(f"[{backend}] building solver (N={N})...")
    t0 = time.time()
    solver = make_solver(backend, model, keys, settings, export_dir)
    print(f"[{backend}] solver built in {time.time() - t0:.1f}s")

    # warm-start initialised to the calibration pose (identical for both backends)
    x_array = np.zeros((nq + nv, N))
    x_array[:nq, :] = q0[:, None]
    u_array = np.zeros((nv, N))
    window = deque(maxlen=N)
    data = pin.Data(model)

    q_list = []
    time_stats = RunningStats()
    first_solve_time = None
    sq_err = {k: [] for k in keys}

    for i, mks in enumerate(frames):
        if i == 0:
            for _ in range(N):
                window.append(mks)
        else:
            window.append(mks)
        marker_meas = np.array(
            [np.hstack([d[k] for k in keys]) for d in window]).T  # (3K, N)

        t0 = time.perf_counter()
        x_array, u_array = solver.solve(
            x_array, u_array, marker_meas, x_array[:, -1], cost_weights, dt)
        elapsed = time.perf_counter() - t0
        if i == 0:
            first_solve_time = elapsed     # cold start, reported separately
        else:
            time_stats.add(elapsed)

        q = np.asarray(x_array[:nq, -1]).flatten()
        q_list.append(q)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for k in keys:
            p = np.asarray(data.oMf[model.getFrameId(k)].translation)
            sq_err[k].append(float(np.sum((p - mks[k]) ** 2)))

        if viz is not None:
            show_frame(viz, model, data, q, mks, keys)
            time.sleep(dt)
        if i % 25 == 0:
            print(f"[{backend}] frame {i}/{len(frames)}")

    return {
        "q": np.array(q_list),
        "time_stats": time_stats,
        "first_solve_time": first_solve_time,
        "rmse_per_marker": {k: float(np.sqrt(np.mean(v))) for k, v in sq_err.items()},
    }


def report(backend, res):
    ts = res["time_stats"]
    glob = float(np.sqrt(np.mean([r ** 2 for r in res["rmse_per_marker"].values()])))
    print(f"\n===== {backend} =====")
    print(f"  tracking RMSE (all markers): {glob * 1000:.2f} mm")
    print(f"  solve time : mean {ts.mean*1e3:.2f}  median {ts.median*1e3:.2f}  "
          f"std {ts.std*1e3:.2f}  max {ts.max*1e3:.2f} ms  (steady state)")
    print(f"  1st (cold) frame: {res['first_solve_time']*1e3:.2f} ms")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(settings):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(cosmik_path, "data",
                                                 "markers_trajectories.csv"))
    p.add_argument("--backends", nargs="+", default=["fatrop", "acados"],
                   choices=["fatrop", "acados"])
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--display-backend", default=None, choices=["fatrop", "acados"])
    p.add_argument("--acados-export-dir",
                   default=os.path.join(cosmik_path, "output", "acados"))
    p.add_argument("--save-dir", default=None)
    return p.parse_args()


def main():
    settings = Settings()
    args = parse_args(settings)
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    frames = load_markers(args.csv)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames; {len(MARKER_NAME_MAP)} mapped markers; "
          f"N={settings.N}, dt={settings.dt}, weights={settings.cost_weights}")

    robot, model, keys, q0 = build_model(frames[0], settings, KEYS_TO_TRACK)
    print(f"Model: nq={model.nq} nv={model.nv}; tracking {len(keys)} markers.")

    backends = list(args.backends)
    if "acados" in backends and ik_module.AcadosOcpSolver is None:
        print("[warn] acados_template not installed -> skipping acados backend.")
        backends.remove("acados")
    if "acados" in backends and "ACADOS_SOURCE_DIR" not in os.environ:
        print("[warn] ACADOS_SOURCE_DIR not set -> skipping acados backend.")
        backends.remove("acados")

    display_backend = args.display_backend or (backends[0] if backends else None)
    results = {}
    for backend in backends:
        viz = None
        if args.display and backend == display_backend:
            viz = make_visualizer(robot, model, keys)
        results[backend] = run_backend(backend, model, keys, frames, q0, settings,
                                       args.acados_export_dir, viz)
        report(backend, results[backend])

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            np.savetxt(os.path.join(args.save_dir, f"q_{backend}.csv"),
                       results[backend]["q"], delimiter=",")

    # backend agreement (equivalence check)
    if "fatrop" in results and "acados" in results:
        qf, qa = results["fatrop"]["q"], results["acados"]["q"]
        n = min(len(qf), len(qa))
        dq = np.abs(qf[:n] - qa[:n])
        print("\n===== fatrop vs acados agreement =====")
        print(f"  joint config |dq|: mean {dq.mean():.2e}  max {dq.max():.2e}")
        speedup = results["fatrop"]["time_stats"].mean / results["acados"]["time_stats"].mean
        print(f"  mean solve-time ratio (fatrop / acados): {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
