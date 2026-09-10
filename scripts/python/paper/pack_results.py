#!/usr/bin/env python3
"""Produce every table the paper needs, from the finished sweeps, in one pass.

Writes results/paper/SUMMARY.txt (readable) and results/paper/summary.json
(machine-readable), covering:

  - whole-body joint RMSE and r per arm per task, mean (std) over participants
  - the lower / trunk / upper segment split
  - marker error split into depth and lateral, and into translation and shape
  - marker, free-flyer and throughput figures

Joint angles are scored against the MoCap modality -- the same markers through
the same model and the same IK -- so the comparison is a clean ablation in which
only the input changes. The dataset's published joint angles come from a
different biomechanical model and are not used anywhere, not even to synchronise.

Marker error is scored against the raw Vicon markers, and every component of it
is an RMS, so that the two decompositions are exact: depth and lateral add in
quadrature to raw, and so do translation and shape. Averaging magnitudes instead
would break both identities and understate the dominant term.

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

DATASET = "/root/workspace/COMFI"

#: Participants dropped from the *marker geometry* section only, never from the
#: joint-angle sections. COMFI's FastSAM export places 3361 at almost exactly
#: twice its true range on all six trials -- depth ratio 1.93 to 1.99, while the
#: image-plane position (43-67 mm) and the body's own size are normal. The error
#: is 2648 mm of pure translation with a 59 mm shape error, so it is a defect in
#: that export's root placement, not something an estimator did. Joint angles are
#: invariant to a per-frame rigid translation and are unaffected, which is why
#: 3361 stays in every other table; the depth and translation columns are not
#: invariant, and one participant at 2.6 m would otherwise set them for the whole
#: arm. Dropped for *all* arms so the comparison stays paired.
#:
#: It is *not* the camera-labelling defect the same participant carries in
#: COMFI's mmpose 2D export, and the two must not be conflated. That one is real
#: -- re-triangulating 3361/Lifting under each candidate labelling scores 101 mm
#: for the pair swap already applied in mmpose_baseline.CAMERA_ID_OVERRIDES
#: (0<->4, 2<->6), against 743 mm for a within-pair swap and 1301 mm as shipped
#: -- but it does not extend to FastSAM. Anchoring the FastSAM export on each of
#: the four cameras gives 2645 / 2679 / 2876 / 2744 mm, and even after fitting a
#: depth scale to absorb the 2x, camera 0 still wins by a wide margin (102 mm
#: against 827 mm for camera 4). The body-centroid bearing, immune to range
#: error, agrees: 3.7 deg for camera 0 against 11.7 for camera 4. The videos are
#: labelled correctly, which is why NLF needs no correction here either.
#:
#: Dividing the camera-frame depth by one scalar (1.93-1.99) restores 3361 to
#: 82-104 mm, which pins the defect to range alone. That correction is NOT
#: applied: the scalar was fitted against mocap, and fitting to the reference
#: would contaminate the result it exists to judge.
MARKER_EXCLUDED_PARTICIPANTS = ("3361",)

ARMS = [
    ("mmpose_0-2", "mmpose 2 cams"),
    ("mmpose_0-2-4-6", "mmpose 4 cams"),
    ("nlf2d_0-2", "NLF-2D tri 2 cams"),
    ("nlf2d_0-2-4-6", "NLF-2D tri 4 cams"),
    ("nlf_0", "NLF-3D 1 cam"),
    ("nlf_0-2", "NLF-3D 2 cams"),
    ("nlf_0-2-4-6", "NLF-3D 4 cams"),
    ("fastsam_0", "FastSAM-3D 1 cam"),
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


class _SkipMarkers(Exception):
    """Raised to skip the marker-geometry block while keeping the joint work."""


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
        depth, lateral, shape, raw, shift = [], [], [], [], []
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

                # Marker geometry, against the raw Vicon markers rather than the
                # mocap_reference run. They are the same markers -- that run is
                # driven by them -- but it also carries Head, REar and LEar,
                # which are stand-ins derived from the Vicon head band and would
                # contribute a definitional offset, not estimation error.
                # Frame indices are shared, so the lag found above applies.
                if participant in MARKER_EXCLUDED_PARTICIPANTS:
                    raise _SkipMarkers
                if participant not in axes:
                    R, _ = load_world_transformation(
                        f"{DATASET}/cam_params/{participant}", 0)
                    axes[participant] = np.asarray(R) @ np.array([0., 0., 1.])
                truth = ev.load_run(f"{DATASET}/mocap/aligned/{participant}/{task}")
                names = sorted(set(run["markers"]) & set(truth["markers"]))
                if names:
                    start, ref_start = max(0, lag), max(0, -lag)
                    P = np.stack([run["markers"][m][start:] for m in names], 1)
                    Q = np.stack([truth["markers"][m][ref_start:] for m in names], 1)
                    k = min(len(P), len(Q))
                    e = P[:k] - Q[:k]
                    e = e[np.isfinite(e).all(2).all(1)]
                    if len(e):
                        ax = axes[participant]
                        along = e @ ax
                        perpendicular = e - along[..., None] * ax
                        translation = e.mean(axis=1, keepdims=True)
                        rms = lambda v: float(np.sqrt((v ** 2).sum(-1).mean()) * 1000)
                        raw.append(rms(e))
                        depth.append(float(np.sqrt((along ** 2).mean()) * 1000))
                        lateral.append(rms(perpendicular))
                        shift.append(rms(translation))
                        shape.append(rms(e - translation))
            except _SkipMarkers:
                continue
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
        entry["translation"] = stat(shift)
        entry["marker_trials"] = len(raw)
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

    add("\n3. MARKER ERROR STRUCTURE (RMS mm, mean (std) across trials)\n")
    add("Two independent splits of the same error. By direction: depth is the")
    add("component along camera 0's optical axis, lateral the rest. By what moves:")
    add("translation is the whole-body shift, shape what is left after removing it.")
    add("Each pair adds in quadrature to raw. Scored against the raw Vicon markers.")
    add(f"Excludes participant(s) {', '.join(MARKER_EXCLUDED_PARTICIPANTS)} from every arm: "
        f"COMFI's FastSAM export")
    add("places that subject at twice its true range, which is a defect in the export's")
    add("root placement, not an estimate. Joint angles are invariant to it and keep all")
    add(f"108 trials; these columns are not, and use "
        f"{data[present[0][0]]['marker_trials']}.\n")
    add(f"{'arm':<20}{'raw':>12}{'depth':>12}{'lateral':>12}{'aniso':>8}"
        f"{'translation':>13}{'shape':>12}")
    for tag, _ in present:
        e = data[tag]
        line = f"{e['label']:<20}"
        for field in ("raw_marker", "depth", "lateral"):
            line += f"{e[field][0]:>7.1f}({e[field][1]:>3.0f})"
        line += f"{e['anisotropy']:>8.2f}"
        for field in ("translation", "shape"):
            line += f"{e[field][0]:>8.1f}({e[field][1]:>3.0f})"
        add(line)

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
