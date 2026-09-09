#!/usr/bin/env python3
"""Produce every table the paper needs, from the finished sweeps, in one pass.

Writes results/paper/SUMMARY.txt (readable) and results/paper/summary.json
(machine-readable), covering:

  - whole-body joint RMSE and r per arm per task, mean (std) over participants
  - the lower / trunk / upper segment split
  - marker error split into depth and lateral, and into translation and shape
  - marker, free-flyer and throughput figures

Everything is scored against the MoCap modality: the dataset's published joint
angles come from a different biomechanical model and are not used anywhere.

    python3 scripts/python/paper/pack_results.py
"""
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

ARMS = [
    ("mmpose_0-2", "mmpose 2 cams"),
    ("mmpose_0-2-4-6", "mmpose 4 cams"),
    ("nlf2d_0-2", "NLF-2D tri 2 cams"),
    ("nlf2d_0-2-4-6", "NLF-2D tri 4 cams"),
    ("nlf_0", "NLF-3D 1 cam"),
    ("nlf_0-2", "NLF-3D 2 cams"),
    ("nlf_0-2-4-6", "NLF-3D 4 cams"),
]
TASKS = ["Screwing", "Polishing", "SideOverhead", "RobotPolishing",
         "RobotWelding", "Lifting"]
LOWER = ("Hip", "Knee", "Ankle")
UPPER = ("Clavicle", "Shoulder", "Elbow")
TRUNK = ("Lumbar", "Cervical")
BONES = [("RSHO", "RELB"), ("RELB", "RWRI"), ("RKNE", "RANK"), ("RASI", "LASI")]


def load_eval():
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_of(name):
    if any(k in name for k in LOWER):
        return "lower"
    if any(k in name for k in UPPER):
        return "upper"
    if any(k in name for k in TRUNK):
        return "trunk"


def stat(values):
    values = [v for v in values if np.isfinite(v)]
    return (float(np.mean(values)), float(np.std(values))) if values else (np.nan, np.nan)


def main():
    from rtcosmik.config_loader import settings
    from rtcosmik.camera.cam_utils import load_world_transformation

    ev = load_eval()
    out_root = Path(settings.output_dir)
    paper = REPO / "results" / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    summaries = {tag: REPO / "results" / "vs_mocap" / f"{tag}.csv" for tag, _ in ARMS}
    present = [(tag, label) for tag, label in ARMS if summaries[tag].exists()]
    if not present:
        raise SystemExit("no rescored summaries under results/vs_mocap/")

    rows = {tag: [r for r in csv.DictReader(open(summaries[tag])) if r["status"] == "ok"]
            for tag, _ in present}
    trials = sorted(set.intersection(
        *({(r["participant"], r["task"]) for r in rows[tag]} for tag, _ in present)))

    data = defaultdict(dict)
    axes = {}
    for tag, label in present:
        per_task = defaultdict(lambda: defaultdict(list))
        groups = defaultdict(lambda: defaultdict(list))
        depth, lateral, shape, raw = [], [], [], []
        for participant, task in trials:
            run_dir = out_root / participant / task / tag
            ref_dir = out_root / participant / task / "mocap_reference"
            if not (run_dir.is_dir() and ref_dir.is_dir()):
                continue
            try:
                run, ref = ev.load_run(str(run_dir)), ev.load_run(str(ref_dir))
                lag, _ = ev.estimate_lag(run, ref)
                a, b = ev.apply_lag(run, ref, lag)
                joints, _ = ev.compare_joint_angles(a, b)
                locked = {j["name"] for j in joints if j.get("locked")}

                header, A, B = a["joint_header"], a["joint_values"], b["joint_values"]
                n = min(len(A), len(B))
                for i, name in enumerate(header):
                    if not name.endswith("[rad]") or name in locked:
                        continue
                    g = group_of(name)
                    if not g:
                        continue
                    d = np.degrees(A[:n, i] - B[:n, i])
                    d = (d + 180) % 360 - 180
                    d = d[np.isfinite(d)]
                    x, y = A[:n, i], B[:n, i]
                    m = np.isfinite(x) & np.isfinite(y)
                    r = (np.corrcoef(x[m], y[m])[0, 1]
                         if m.sum() > 2 and x[m].std() > 1e-9 and y[m].std() > 1e-9
                         else np.nan)
                    rmse = float(np.sqrt((d ** 2).mean()))
                    per_task[task][participant].append((rmse, r))
                    groups[g][participant].append((rmse, r))
                    groups["ALL"][participant].append((rmse, r))

                # marker geometry
                if participant not in axes:
                    R, _ = load_world_transformation(
                        f"/root/workspace/COMFI/cam_params/{participant}", 0)
                    axes[participant] = np.asarray(R) @ np.array([0., 0., 1.])
                names = sorted(set(a["markers"]) & set(b["markers"]))
                if names:
                    P = np.stack([a["markers"][m] for m in names], 1)
                    Q = np.stack([b["markers"][m] for m in names], 1)
                    k = min(len(P), len(Q))
                    e = P[:k] - Q[:k]
                    ok = np.isfinite(e).all(2).all(1)
                    e = e[ok]
                    if len(e):
                        ax = axes[participant]
                        depth.append(np.abs(e @ ax).mean() * 1000)
                        lateral.append(np.linalg.norm(
                            e - (e @ ax)[..., None] * ax, axis=2).mean() * 1000)
                        raw.append(np.linalg.norm(e, axis=2).mean() * 1000)
                        shape.append(np.linalg.norm(
                            e - e.mean(axis=1, keepdims=True), axis=2).mean() * 1000)
            except Exception:
                continue

        entry = {"label": label, "tasks": {}, "groups": {}}
        for task in TASKS:
            per = [np.mean([v[0] for v in vals]) for vals in per_task[task].values()]
            rr = [np.mean([v[1] for v in vals]) for vals in per_task[task].values()]
            entry["tasks"][task] = {"rmse": stat(per), "r": stat(rr)}
        for g in ("lower", "trunk", "upper", "ALL"):
            per = [np.mean([v[0] for v in vals]) for vals in groups[g].values()]
            rr = [np.nanmean([v[1] for v in vals]) for vals in groups[g].values()]
            entry["groups"][g] = {"rmse": stat(per), "r": stat(rr)}
        entry["depth"] = stat(depth)
        entry["lateral"] = stat(lateral)
        entry["raw_marker"] = stat(raw)
        entry["shape"] = stat(shape)
        entry["anisotropy"] = (entry["depth"][0] / entry["lateral"][0]
                               if entry["lateral"][0] else np.nan)
        for field in ("marker_mm", "freeflyer_mm", "fps", "ik_ms_median"):
            entry[field] = stat([float(r[field]) for r in rows[tag] if r.get(field)])
        data[tag] = entry

    lines = []
    add = lines.append
    add("=" * 96)
    add(f"RT-COSMIK modality study -- {len(trials)} trials, 18 participants, 6 tasks, "
        f"29 scored DoF")
    add(f"Reference: mocap markers through the same model and IK. "
        f"Filter: order {settings.order}, cutoff {settings.cutoff_freq:g} Hz.")
    add("=" * 96)

    add("\n1. WHOLE-BODY JOINT RMSE (deg), mean (std) across participants\n")
    add(f"{'arm':<20}" + "".join(f"{t[:11]:>13}" for t in TASKS) + f"{'OVERALL':>14}{'r':>8}")
    for tag, _ in present:
        e = data[tag]
        line = f"{e['label']:<20}"
        for t in TASKS:
            m, s = e["tasks"][t]["rmse"]
            line += f"{m:>8.2f}({s:>3.1f})"
        m, s = e["groups"]["ALL"]["rmse"]
        line += f"{m:>9.2f}({s:>3.1f}){e['groups']['ALL']['r'][0]:>8.3f}"
        add(line)

    add("\n2. SEGMENT GROUPS (deg / r)\n")
    add(f"{'arm':<20}{'lower':>16}{'trunk':>16}{'upper':>16}{'all':>16}")
    for tag, _ in present:
        e = data[tag]
        line = f"{e['label']:<20}"
        for g in ("lower", "trunk", "upper", "ALL"):
            line += f"{e['groups'][g]['rmse'][0]:>10.2f}/{e['groups'][g]['r'][0]:<5.2f}"
        add(line)

    add("\n3. MARKER ERROR STRUCTURE (mm)\n")
    add(f"{'arm':<20}{'raw':>10}{'depth':>10}{'lateral':>10}{'aniso':>8}"
        f"{'shape':>10}{'transl.share':>14}")
    for tag, _ in present:
        e = data[tag]
        share = (1 - e["shape"][0] / e["raw_marker"][0]) * 100 if e["raw_marker"][0] else np.nan
        add(f"{e['label']:<20}{e['raw_marker'][0]:>10.1f}{e['depth'][0]:>10.1f}"
            f"{e['lateral'][0]:>10.1f}{e['anisotropy']:>8.2f}{e['shape'][0]:>10.1f}"
            f"{share:>13.0f}%")

    add("\n4. THROUGHPUT AND SOLVER (from the sweep; see Table I for clean timings)\n")
    add(f"{'arm':<20}{'fps':>10}{'IK ms':>10}{'marker mm':>12}{'freeflyer mm':>14}")
    for tag, _ in present:
        e = data[tag]
        add(f"{e['label']:<20}{e['fps'][0]:>10.1f}{e['ik_ms_median'][0]:>10.2f}"
            f"{e['marker_mm'][0]:>12.1f}{e['freeflyer_mm'][0]:>14.1f}")

    text = "\n".join(lines)
    (paper / "SUMMARY.txt").write_text(text + "\n")
    json.dump({k: v for k, v in data.items()}, open(paper / "summary.json", "w"),
              indent=2, default=float)
    print(text)
    print(f"\nwritten to {paper}/SUMMARY.txt and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
