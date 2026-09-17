#!/usr/bin/env python3
"""Design-choice studies on the proposed pipeline: IK type, MHE horizon, filter.

E3  IK type        sample-by-sample QP ("sbs") against the MHE
E4  MHE horizon    N = 3, 5, 7, 10, 15, 20
E5  marker filter  none; 2nd order at 6, 8, 10 Hz; 4th order at 3, 4, 5, 8, 10 Hz

All three change only what happens after NLF, so nothing is re-estimated: the
NLF-3D 4-camera sweep saved every frame's per-camera NLF output
(``sweep.py --views-cache``), and each variant replays it through the same
reconstruction -> world transform -> filter -> IK as ``sweep.run_nlf``. The
baseline variant (MHE, N = 7, 4th order at 5 Hz) is the shipped configuration and
must reproduce the sweep's own run; ``check`` reports how closely it does.

Per trial and variant, written to ``<out>/studies/<variant>.csv``:

* accuracy against the mocap reference, lag-compensated (as everywhere) and at
  zero lag -- the filter study needs the latter, since compensation hides delay
* residual lag (frames), and for filters their analytic group delay at 1 Hz
* jitter: median |frame-to-frame change| of the scored DoF (deg), and median RMS
  jerk (deg/s^3)
* joint-limit activity: share of frames with a scored DoF within 0.5 deg of a
  bound, and with one beyond a bound by more than 0.06 deg (violation)
* per-frame solve time p50 / p95 / max (ms, calibration frame excluded) and
  failed frames (solver exception or non-finite result)

Solve times are measured with several variants running in parallel on the CPU
(``--workers``): good for comparing variants, not a clean latency benchmark.

Each horizon other than the shipped one needs its own generated OCP. They are
generated first, one at a time, into ``ocp/studies/N<k>/`` -- never into the
pipeline's own artefact folder, and never by parallel replays racing each other.

    python3 scripts/python/paper/ik_filter_studies.py \\
        --views results/campaign/views --output-dir output/campaign \\
        --out results/campaign/paper
"""
import argparse
import csv
import importlib.util
import logging
import os
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
SOURCE_TAG = "nlf_0-2-4-6"
FS = 40.0

#: name -> (study, ik_type, N, filter order or None, cutoff Hz)
VARIANTS = OrderedDict([
    ("mhe_N7_o4c5", ("baseline", "mhe", 7, 4, 5.0)),
    ("sbs_o4c5", ("E3", "sbs", 7, 4, 5.0)),
    ("mhe_N3_o4c5", ("E4", "mhe", 3, 4, 5.0)),
    ("mhe_N5_o4c5", ("E4", "mhe", 5, 4, 5.0)),
    ("mhe_N10_o4c5", ("E4", "mhe", 10, 4, 5.0)),
    ("mhe_N15_o4c5", ("E4", "mhe", 15, 4, 5.0)),
    ("mhe_N20_o4c5", ("E4", "mhe", 20, 4, 5.0)),
    ("mhe_N7_nofilter", ("E5", "mhe", 7, None, None)),
    ("mhe_N7_o2c6", ("E5", "mhe", 7, 2, 6.0)),
    ("mhe_N7_o2c8", ("E5", "mhe", 7, 2, 8.0)),
    ("mhe_N7_o2c10", ("E5", "mhe", 7, 2, 10.0)),
    ("mhe_N7_o4c8", ("E5", "mhe", 7, 4, 8.0)),
    ("mhe_N7_o4c10", ("E5", "mhe", 7, 4, 10.0)),
    ("mhe_N7_o4c4", ("E5", "mhe", 7, 4, 4.0)),
    ("mhe_N7_o4c3", ("E5", "mhe", 7, 4, 3.0)),
])
LIMIT_NEAR_RAD, LIMIT_VIOLATION_RAD = np.radians(0.5), 1e-3
FIELDS = ["variant", "study", "ik_type", "N", "filter", "participant", "task", "frames",
          "joint_rmse_deg", "joint_rmse_zero_lag_deg", "upper_rmse_deg", "lower_rmse_deg",
          "trunk_rmse_deg", "shoulder_flip_pct", "lag_frames", "filter_delay_1hz_ms",
          "jitter_deg", "jerk_rms_deg_s3", "near_limit_pct", "violation_pct",
          "solve_ms_p50", "solve_ms_p95", "solve_ms_max", "failed_frames"]


def tag(variant):
    return f"study_{variant}"


def ocp_root(horizon, settings):
    """Artefact root for a horizon: the pipeline's own for the shipped N."""
    return None if horizon == settings.N else REPO / "ocp" / "studies" / f"N{horizon}"


def prepare_ocp(horizon):
    """Generate the acados OCP for one study horizon, unless it is up to date."""
    from rtcosmik.config_loader import settings
    root = ocp_root(horizon, settings)
    if root is None:
        return f"N={horizon}: the pipeline's own artefact"
    os.environ["RTCOSMIK_OCP_DIR"] = str(root)
    settings.N = horizon
    spec = importlib.util.spec_from_file_location(
        "run_ocp_codegen", REPO / "scripts" / "python" / "core" / "run_ocp_codegen.py")
    codegen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codegen)
    model, keys = codegen.structural_model()
    if not codegen.do_check(["acados"], [settings.mhe_profile], model, keys):
        return f"N={horizon}: up to date in {root}"
    codegen.generate_acados(model, keys, settings.mhe_profile)
    return f"N={horizon}: generated in {root}"


def filter_delay_ms(order, cutoff):
    """Group delay of the causal Butterworth at 1 Hz, in ms."""
    if order is None:
        return 0.0
    from scipy import signal
    b, a = signal.butter(order, cutoff, "lowpass", fs=FS)
    _, delay = signal.group_delay((b, a), w=[1.0], fs=FS)
    return float(delay[0] / FS * 1000.0)


def replay(job):
    """One variant on one trial: write the run folder and return timing and failures."""
    runs_root, views_root, participant, task, variant = job
    import yaml
    from rtcosmik.config_loader import settings
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.saver.csv_saver import CSVSaver

    study, ik_type, horizon, order, cutoff = VARIANTS[variant]
    root = ocp_root(horizon, settings)
    if root is not None:
        os.environ["RTCOSMIK_OCP_DIR"] = str(root)
    settings.ik_type, settings.N = ik_type, horizon
    if order is not None:
        settings.order, settings.cutoff_freq = order, cutoff
    from rtcosmik.paper.nlf_views import replay_markers
    frames = replay_markers(views_root / participant / f"{task}.npz", DATASET / "cam_params" / participant)
    marker_filter = MarkerFilter(len(settings.marker_names), settings) if order is not None else None

    meta = yaml.safe_load((DATASET / "metadata" / f"{participant}.yaml").read_text())
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    run_dir = runs_root / participant / task / tag(variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    saver = CSVSaver(str(run_dir), markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    solve_ms, failed, q_last = [], 0, None
    for index, (frame, p3d) in enumerate(frames):
        points = marker_filter(p3d) if marker_filter is not None else p3d
        mks = dict(zip(settings.marker_names, points))
        t0 = time.perf_counter()
        try:
            q = solver.solve(mks)
            ok = np.all(np.isfinite(q))
        except Exception:
            q, ok = None, False
        if index:
            solve_ms.append((time.perf_counter() - t0) * 1e3)
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
    return {"solve_ms_p50": float(np.nanpercentile(solve_ms, 50)),
            "solve_ms_p95": float(np.nanpercentile(solve_ms, 95)),
            "solve_ms_max": float(np.nanmax(solve_ms)), "failed_frames": failed}


def smoothness_and_limits(run_dir, scored):
    """Jitter, jerk and joint-limit activity over the scored DoF of one run."""
    import example_robot_data as robex
    import pandas as pd
    angles = pd.read_csv(run_dir / "joint_angles.csv")
    model = robex.human.HumanLoader(height=1.7, weight=70.0, gender="m").robot.model
    names = list(angles.columns)
    columns = [names.index(n) for n in scored]
    q = angles.to_numpy(float)
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit
    x = q[:, columns]
    lo, hi = lower[columns], upper[columns]
    dq = np.degrees(np.abs(np.diff(x, axis=0)))
    jerk = np.degrees(np.diff(x, n=3, axis=0)) * FS ** 3
    near = ((x - lo) < LIMIT_NEAR_RAD) | ((hi - x) < LIMIT_NEAR_RAD)
    beyond = (x < lo - LIMIT_VIOLATION_RAD) | (x > hi + LIMIT_VIOLATION_RAD)
    return {"jitter_deg": float(np.median(dq)),
            "jerk_rms_deg_s3": float(np.median(np.sqrt((jerk ** 2).mean(axis=0)))),
            "near_limit_pct": float(100 * near.any(axis=1).mean()),
            "violation_pct": float(100 * beyond.any(axis=1).mean())}


def score(job):
    """All metrics for one replayed variant on one trial."""
    runs_root, participant, task, variant, timing = job
    import trial_metrics as tm
    ev = tm.load_eval()
    run_dir = runs_root / participant / task / tag(variant)
    ref_dir = runs_root / participant / task / REFERENCE_TAG
    truth = DATASET / "mocap" / "aligned" / participant / task
    dofs, trial = tm.trial_metrics(ev, run_dir, ref_dir, truth, participant)
    zero_dofs, _ = tm.trial_metrics(ev, run_dir, ref_dir, truth, participant, lag=0)
    by_group = lambda g: float(np.mean([d["rmse_deg"] for d in dofs if d["group"] == g]))
    study, ik_type, horizon, order, cutoff = VARIANTS[variant]
    row = {"variant": variant, "study": study, "ik_type": ik_type, "N": horizon,
           "filter": "none" if order is None else f"order {order}, {cutoff:g} Hz",
           "participant": participant, "task": task, "frames": trial["frames"],
           "joint_rmse_deg": trial["joint_rmse_deg"],
           "joint_rmse_zero_lag_deg": float(np.mean([d["rmse_deg"] for d in zero_dofs])),
           "upper_rmse_deg": by_group("upper"), "lower_rmse_deg": by_group("lower"),
           "trunk_rmse_deg": by_group("trunk"), "shoulder_flip_pct": trial["shoulder_flip_pct"],
           "lag_frames": trial["lag_frames"], "filter_delay_1hz_ms": filter_delay_ms(order, cutoff)}
    row.update(smoothness_and_limits(run_dir, [d["dof"] + "[rad]" for d in dofs]))
    row.update(timing)
    return row


def check(runs_root, trials):
    """How closely the baseline replay reproduces the sweep's NLF-3D 4-camera run."""
    import pandas as pd
    worst = 0.0
    for participant, task in trials:
        a = runs_root / participant / task / SOURCE_TAG / "joint_angles.csv"
        b = runs_root / participant / task / tag("mhe_N7_o4c5") / "joint_angles.csv"
        if not (a.exists() and b.exists()):
            continue
        A, B = pd.read_csv(a).to_numpy(float), pd.read_csv(b).to_numpy(float)
        if A.shape != B.shape:
            print(f"  check {participant}/{task}: {A.shape} rows vs {B.shape}")
            worst = np.inf
            continue
        worst = max(worst, float(np.abs(A - B).max()))
    print(f"baseline replay vs sweep run: max |dq| = {worst:.3g}", flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True, help="the sweep's --views-cache folder")
    ap.add_argument("--output-dir", type=Path, required=True, help="where run folders live")
    ap.add_argument("--out", type=Path, required=True, help="paper results folder")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    trials = sorted((p.parent.name, p.stem) for p in args.views.glob("*/*.npz")
                    if (args.output_dir / p.parent.name / p.stem / REFERENCE_TAG).is_dir())
    jobs = [(args.output_dir, args.views, p, t, v) for p, t in trials for v in VARIANTS
            if not (args.output_dir / p / t / tag(v) / "joint_angles.csv").exists()]
    print(f"{len(trials)} trials x {len(VARIANTS)} variants; {len(jobs)} replays to run", flush=True)

    # A fresh process per job: variants set different settings and load
    # different compiled solvers, and neither should leak into the next job.
    import multiprocessing
    horizons = sorted({VARIANTS[v][2] for v in VARIANTS if VARIANTS[v][1] == "mhe"})
    if jobs:
        with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
            for message in pool.imap(prepare_ocp, horizons):
                print(f"  OCP {message}", flush=True)
    timings = {}
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for job, timing in zip(jobs, pool.imap(replay, jobs)):
            timings[(job[2], job[3], job[4])] = timing
            print(f"  {job[2]}/{job[3]} {job[4]}: {timing['solve_ms_p50']:.2f} ms p50, "
                  f"{timing['failed_frames']} failed", flush=True)
    # Replays from an earlier, interrupted run keep their runs but not their timing.
    empty = {"solve_ms_p50": np.nan, "solve_ms_p95": np.nan, "solve_ms_max": np.nan, "failed_frames": np.nan}
    check(args.output_dir, trials)

    score_jobs = [(args.output_dir, p, t, v, timings.get((p, t, v), empty))
                  for p, t in trials for v in VARIANTS]
    rows = {v: [] for v in VARIANTS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(score, score_jobs):
            rows[row["variant"]].append(row)
    target = args.out / "studies"
    target.mkdir(parents=True, exist_ok=True)
    for variant, variant_rows in rows.items():
        with open(target / f"{variant}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(variant_rows)
    print(f"{sum(len(r) for r in rows.values())} study rows -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
