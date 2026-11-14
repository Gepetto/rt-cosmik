#!/usr/bin/env python3
"""Produce the NLF arms that differ from the 4-camera run only after NLF, by replay.

The NLF-3D 4-camera sweep records every camera's NLF output
(``sweep.py --views-cache``). The other NLF arms -- fewer cameras, and NLF's 2D
keypoints triangulated instead of its 3D fused -- are rebuilt from that
recording on the CPU (``rtcosmik.paper.nlf_views``) through the same filter and
IK as ``sweep.run_nlf``, and written exactly like a sweep: run folders plus one
summary row per trial. Every camera configuration therefore sees identical NLF
output, so the camera and reconstruction effects are not mixed with inference
noise. Replay reproduces direct runs to 0.3-0.4 mm median marker difference
(at most 0.08 deg of joint RMSE per trial, participant 1012).

A replay has no meaningful throughput, so ``fps`` is left empty: the timing of
these arms comes from direct runs on a few participants (``run_campaign.sh``).
Resumable like the sweep: trials already in a summary are skipped.

    python3 scripts/python/paper/replay_nlf_arms.py --views results/campaign/views \\
        --output-dir output/campaign --results results/campaign
"""
import argparse
import csv
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
#: tag -> (sweep arm name, reconstruction, cameras)
ARMS = OrderedDict([
    ("nlf_0", ("nlf", "fuse3d", [0])),
    ("nlf_0-2", ("nlf", "fuse3d", [0, 2])),
    ("nlf_0-4", ("nlf", "fuse3d", [0, 4])),
    ("nlf2d_0-2-4-6", ("nlf2d", "tri2d", [0, 2, 4, 6])),
    ("nlf2d_0-2", ("nlf2d", "tri2d", [0, 2])),
    ("nlf2d_0-4", ("nlf2d", "tri2d", [0, 4])),
])


def replay_trial(dataset, views_path, out_dir, participant, task, arm, settings, repo=REPO):
    """One arm on one recorded trial; returns a sweep.FIELDS row."""
    import yaml
    import sweep
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.landmark_maps import checked_recording
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.provenance import recorded_estimator, start_run, update_run_info
    from rtcosmik.saver.csv_saver import CSVSaver

    dataset, out_dir = Path(dataset), Path(out_dir)
    cameras = list(arm["cameras"])
    from rtcosmik.paper import require_paper_evaluation
    require_paper_evaluation(settings)
    landmark_map = checked_recording(views_path, participant, settings)
    start_run(out_dir, settings, dict(arm, participant=participant, task=task,
                                      views=str(views_path), dataset=str(dataset)), repo=repo,
              landmark_map=landmark_map, estimator=recorded_estimator(views_path))
    row = dict.fromkeys(sweep.FIELDS, "")
    row.update(arm=arm["name"], participant=participant, task=task,
               cameras="-".join(map(str, cameras)), n_horizon=settings.N)
    frames = replay_markers(views_path, dataset / "cam_params" / participant, cameras,
                            arm["reconstruction"])
    meta = yaml.safe_load((dataset / "metadata" / f"{participant}.yaml").read_text())
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    marker_filter = MarkerFilter(len(settings.marker_names), settings)
    saver = CSVSaver(str(out_dir), markers_header=["Frame_0"] + list(settings.marker_names),
                     joint_angles_header=list(settings.joint_angles_names))
    ik_ms = []
    for index, (frame, p3d) in enumerate(frames):
        mks = dict(zip(settings.marker_names, marker_filter(p3d)))
        t0 = time.perf_counter()
        q = solver.solve(mks)
        if index:
            ik_ms.append((time.perf_counter() - t0) * 1e3)
        markers = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            markers[f"{name}_x"], markers[f"{name}_y"], markers[f"{name}_z"] = map(float, mks[name])
        saver.save_markers(markers)
        saver.save_joint_angles(OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
    saver.close()
    row.update(frames=len(frames),
               ik_ms_median=round(float(np.median(ik_ms)), 2) if ik_ms else "",
               ik_ms_p95=round(float(np.percentile(ik_ms, 95)), 2) if ik_ms else "")
    update_run_info(out_dir, frames=len(frames))
    from rtcosmik.paper.scoring import score_run
    row.update(score_run(out_dir, dataset, participant, task, settings))
    row["status"] = "ok"
    return row


def replay(job):
    """Pool worker: one arm on one trial, failures reported in the row."""
    import sweep
    from rtcosmik.config_loader import settings

    runs_root, views_root, participant, task, tag = job
    name, reconstruction, cameras = ARMS[tag]
    arm = {"name": name, "tag": tag, "reconstruction": reconstruction, "cameras": cameras}
    try:
        row = replay_trial(DATASET, views_root / participant / f"{task}.npz",
                           runs_root / participant / task / tag, participant, task, arm, settings)
    except Exception as exc:
        row = dict.fromkeys(sweep.FIELDS, "")
        row.update(arm=name, participant=participant, task=task,
                   cameras="-".join(map(str, cameras)), n_horizon=settings.N,
                   status=f"{type(exc).__name__}: {exc}"[:200])
    return tag, row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True, help="the sweep's --views-cache folder")
    ap.add_argument("--output-dir", type=Path, required=True, help="where run folders go")
    ap.add_argument("--results", type=Path, required=True, help="where the summaries <tag>.csv go")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import multiprocessing
    import sweep
    from rtcosmik.provenance import require_clean
    require_clean()
    trials = sorted((p.parent.name, p.stem) for p in args.views.glob("*/*.npz"))
    done = {}
    for tag in ARMS:
        path = args.results / f"{tag}.csv"
        done[tag] = ({(r["participant"], r["task"]) for r in csv.DictReader(open(path))}
                     if path.exists() else set())
    jobs = [(args.output_dir, args.views, p, t, tag) for tag in ARMS for p, t in trials
            if (p, t) not in done[tag]]
    print(f"{len(trials)} recorded trials x {len(ARMS)} arms; {len(jobs)} replays to run", flush=True)

    handles = {}
    for tag in ARMS:
        path = args.results / f"{tag}.csv"
        fresh = not path.exists()
        handles[tag] = open(path, "a", newline="")
        writer = csv.DictWriter(handles[tag], fieldnames=sweep.FIELDS)
        if fresh:
            writer.writeheader()
    failed = 0
    # A fresh process per replay, as the studies do: nothing leaks between trials.
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for tag, row in pool.imap_unordered(replay, jobs):
            csv.DictWriter(handles[tag], fieldnames=sweep.FIELDS).writerow(row)
            handles[tag].flush()
            if row["status"] != "ok":
                failed += 1
                print(f"  {tag} {row['participant']}/{row['task']}: {row['status']}", flush=True)
    for handle in handles.values():
        handle.close()
    print(f"replays done, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
