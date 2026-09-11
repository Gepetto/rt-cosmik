#!/usr/bin/env python3
"""Does one generated OCP really serve every subject?

Emulates the online situation this parameterization exists for: a session in
which the person in front of the cameras changes, so the calibrated model changes
between trials. Previously that forced a regeneration of the OCP -- 20-40 s of
compile in the middle of a live run. Now it should be a parameter update.

For each participant the script builds the calibrated model the ordinary way,
then solves the same marker data twice:

    baked    a solver generated from that subject's own model  (today's behaviour)
    generic  one solver generated from the structural seed, with the subject's
             geometry set as parameters                        (the new path)

They must agree to solver tolerance. The script then switches subjects back and
forth on the *same* generic solver to confirm the switch is stateless and that
nothing is regenerated on the way.

    python3 tests/benchmark/validate_ocp_params.py --participants 1012 1118 1508
    python3 tests/benchmark/validate_ocp_params.py --backend fatrop
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pinocchio as pin
import example_robot_data as robex

from rtcosmik.config_loader import settings
from rtcosmik.human_model.model_utils import (
    scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space)
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.ik import ocp_model
from benchmark_mhe_ik_backends import MARKER_NAME_MAP

DEFAULT_ROOT = "/home/msabbah/Desktop/comfi-examples/downloads"


def load_frames(root, participant, task, count):
    path = os.path.join(root, "mocap", "mocap", "aligned", participant, task,
                        "markers_trajectories.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, nrows=count)
    frames = []
    for _, row in df.iterrows():
        frame = {}
        for cos, csv in MARKER_NAME_MAP.items():
            try:
                frame[cos] = np.array([row[f"{csv}_X[mm]"], row[f"{csv}_Y[mm]"],
                                       row[f"{csv}_Z[mm]"]], float) / 1000.0
            except KeyError:
                pass
        frames.append(frame)
    return frames


def subject_meta(root, participant):
    import yaml
    with open(os.path.join(root, "metadata", "metadata", f"{participant}.yaml")) as fh:
        meta = yaml.safe_load(fh)
    return float(meta["height"]), float(meta["weight"]), str(meta["gender"])[0].lower()


def calibrate(frames, height, weight, gender, keys):
    """The ordinary per-subject model build: scale, register, IK, recalibrate."""
    mks0 = frames[0]
    model = robex.human.HumanLoader(height=height, weight=weight,
                                    gender=gender).robot.model
    model = scale_human_model(model, mks0, gender=gender, subject_height=height)
    model = mks_registration(model, mks0, gender=gender, subject_height=height)
    omega = {k: 1.0 for k in keys}
    q0 = np.asarray(RT_IK(model, mks0, pin.neutral(model), keys, settings.dt,
                          omega).solve_ik_sample_casadi()).flatten()
    model = recalibrate_marker_frames_in_joint_space(
        model, q0, mks0, list(MARKER_NAME_MAP.keys()))
    return model, q0


def run_window(solver, model, frames, keys, q0, n_steps):
    """Solve a short sliding window, returning the q at each step."""
    N = settings.N
    nx, nu = model.nq + model.nv, model.nv
    X = np.zeros((nx, N))
    X[:model.nq, :] = np.tile(np.asarray(q0).reshape(-1, 1), (1, N))
    U = np.zeros((nu, N))
    window = [frames[0]] * N
    out = []
    for step in range(n_steps):
        window = window[1:] + [frames[min(step, len(frames) - 1)]]
        meas = np.array([np.hstack([f[k] for k in keys]) for f in window]).T
        X, U = solver.solve(X, U, meas, X[:, -1], settings.cost_weights, settings.dt)
        X = np.asarray(X)
        U = np.asarray(U)
        out.append(np.array(X[:model.nq, -1]).flatten())
    return np.array(out)


def make_solver(backend, model, keys, build, export_dir=None):
    """Build or load a solver.

    ``build=True`` regenerates for this specific subject -- the reference the
    shared solver is checked against. It must write somewhere else: generating
    into the shared artefact directory would overwrite the very thing under test.
    """
    directory = export_dir or ocp_model.backend_dir(backend, settings)
    if backend == "acados":
        return RT_SWIKA_ACADOS(
            model, keys, settings.N, settings.dt, build=build,
            export_dir=directory,
            acados_source_dir=settings.acados_source_dir,
            solver_options=ocp_model.profile_options("acados", settings.mhe_profile))
    return RT_SWIKA_FATROP(
        model, keys, settings.N,
        solver_options=ocp_model.profile_options("fatrop", settings.mhe_profile),
        code=("c" if not build else "python"), export_dir=directory)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="COMFI dataset root")
    ap.add_argument("--participants", nargs="+",
                    default=["1012", "1118", "1508", "4279"])
    ap.add_argument("--task", default="Lifting")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--backend", choices=["fatrop", "acados"], default="acados")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |dq| in rad allowed between baked and generic")
    args = ap.parse_args()

    keys = list(settings.keys_to_track_list)

    # The generic solver: generated once, from the structural seed, never rebuilt.
    print(f"Loading the generated {args.backend} OCP "
          f"(built from the structural seed, never from a subject) ...")
    seed_model = ocp_model.build_structural_model(settings)
    started = time.time()
    generic = make_solver(args.backend, seed_model, keys, build=False)
    print(f"  loaded in {time.time()-started:.2f} s, "
          f"{generic.n_params} geometry parameters\n")

    # Per-subject reference solvers go somewhere disposable, so the shared
    # artefact they are compared against is never overwritten.
    reference_dir = tempfile.mkdtemp(prefix="ocp_reference_")
    print(f"per-subject reference solvers -> {reference_dir}\n")

    results, failures = [], []
    calibrated = {}
    for participant in args.participants:
        frames = load_frames(args.root, participant, args.task, args.steps + settings.N)
        if not frames:
            print(f"{participant}: no {args.task} data, skipped")
            continue
        height, weight, gender = subject_meta(args.root, participant)
        model, q0 = calibrate(frames, height, weight, gender, keys)
        calibrated[participant] = (model, q0, frames)

        started = time.time()
        baked = make_solver(args.backend, model, keys, build=True,
                            export_dir=os.path.join(reference_dir, participant))
        build_s = time.time() - started

        started = time.time()
        generic.set_model_params(model)
        set_ms = (time.time() - started) * 1000

        q_baked = run_window(baked, model, frames, keys, q0, args.steps)
        q_generic = run_window(generic, model, frames, keys, q0, args.steps)
        worst = float(np.abs(q_baked - q_generic).max())
        ok = worst <= args.tol
        results.append((participant, height, build_s, set_ms, worst, ok))
        if not ok:
            failures.append(participant)
        print(f"{participant} (h={height:.2f} m): regenerate {build_s:6.1f} s vs "
              f"set params {set_ms:.3f} ms | max |dq| = {worst:.2e} rad "
              f"{'OK' if ok else 'MISMATCH'}")

    # Switching subjects mid-session, on one solver, must be stateless.
    print("\nSwitching subjects on the same solver (the online case):")
    order = [p for p in args.participants if p in calibrated]
    if len(order) >= 2:
        a, b = order[0], order[1]
        seq = [a, b, a, b]
        first = {}
        consistent = True
        for participant in seq:
            model, q0, frames = calibrated[participant]
            generic.set_model_params(model)
            q = run_window(generic, model, frames, keys, q0, args.steps)
            if participant in first:
                delta = float(np.abs(first[participant] - q).max())
                consistent &= delta <= 1e-12
                print(f"  {participant} again: max |dq| vs its first pass = {delta:.2e}")
            else:
                first[participant] = q
                print(f"  {participant}: first pass")
        print(f"  switching is stateless: {consistent}")
        if not consistent:
            failures.append("switching")

    print()
    if results:
        arr = np.array([[r[2], r[3]] for r in results])
        print(f"regenerating would cost {arr[:,0].sum():.0f} s across "
              f"{len(results)} subjects; setting parameters cost "
              f"{arr[:,1].sum():.1f} ms in total.")
    if failures:
        print(f"FAILED: {', '.join(str(f) for f in failures)}")
        return 1
    print("All subjects agree with their own baked solver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
