#!/usr/bin/env python3
"""Fusion rules under lost views and outliers, by replay of recorded views.

Each trial's recorded NLF output is replayed through the shipped fusion, filter
and moving-horizon IK under every fusion rule, clean and perturbed:

``mask``     every view is dropped, independently, for stretches of frames
             (mean ``STRETCH_S``), a share ``level`` of the time.
``outlier``  in one view per trial, one limb is displaced by ``level`` metres in
             a random direction for stretches of frames (``OUTLIER_RATE`` of the
             time). Sigma is left as recorded: the outlier is not announced.

A trial's perturbation depends only on the trial and the condition, so every
rule sees the same one. The levels of ``GRID`` are run on the validation
participants (folds 0 and 1); per perturbation, the mildest level that raises
the unweighted mean's causal error by ``MATERIAL_DEG`` there (the strongest if
none does) is then run on the other twelve participants.

    python3 scripts/python/paper/study_fusion_robustness.py --views /root/workspace/oof/views \\
        --output-dir /root/workspace/oof/t10/runs --out /root/workspace/oof/t10/robustness
"""
import argparse
import csv
import json
import logging
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from rtcosmik.folds import FOLDS, PARTICIPANTS
from rtcosmik.paper import fusion_validation as fv
from rtcosmik.paper import statistics as st
from rtcosmik.paper.scoring import ROW_FIELDS
from rtcosmik.triangulation.triangulation import FUSION_RULES

TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
ROBOT_TASKS = ("RobotPolishing", "RobotWelding")
FS = 40.0
WORKERS = 10
VALIDATION = FOLDS[0] + FOLDS[1]
HELD_OUT = tuple(p for p in PARTICIPANTS if p not in VALIDATION)
STRETCH_S = 1.0
OUTLIER_RATE = 0.2
MATERIAL_DEG = 0.5
BASELINE = "mean"
#: condition -> (perturbation, level), mildest first within a perturbation
GRID = OrderedDict([("clean", (None, 0.0)),
                    ("mask10", ("mask", 0.10)), ("mask25", ("mask", 0.25)), ("mask50", ("mask", 0.50)),
                    ("out10", ("outlier", 0.10)), ("out30", ("outlier", 0.30)), ("out100", ("outlier", 1.00))])
FIELDS = ["phase", "condition", "rule", "participant", "task"] + ROW_FIELDS


def tag(condition, rule):
    return f"robust_{condition}_{rule}"


def perturbation(condition, participant, task, names):
    """The edit of a recording's arrays for ``condition``, the same for every rule."""
    kind, level = GRID[condition]
    if kind is None:
        return None
    length = STRETCH_S * FS

    def edit(keypoints, poses, sigma):
        rng = fv.seeded(participant, task, condition)
        if kind == "mask":
            return fv.mask_views(keypoints, poses, sigma, level, length, rng)
        return keypoints, fv.inject_outliers(poses, names, level, OUTLIER_RATE, length, rng)[0], sigma
    return edit


def replay(job):
    runs_root, views_root, dataset, participant, task, condition, rule = job
    import yaml
    from rtcosmik.config_loader import settings
    from rtcosmik.filtering.iir import MarkerFilter
    from rtcosmik.landmark_maps import checked_recording
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.provenance import recorded_estimator, start_run, update_run_info
    from rtcosmik.saver.csv_saver import CSVSaver

    run_dir = runs_root / participant / task / tag(condition, rule)
    info = run_dir / "run_info.json"
    if info.exists() and "solver_failures" in json.loads(info.read_text()):
        return
    recording = views_root / participant / f"{task}.npz"
    landmark_map = checked_recording(recording, participant, settings)
    names = list(settings.marker_names)
    frames = replay_markers(recording, dataset / "cam_params" / participant, fusion=rule,
                            edit=perturbation(condition, participant, task, names))
    marker_filter = MarkerFilter(len(names), settings)
    meta = yaml.safe_load((dataset / "metadata" / f"{participant}.yaml").read_text())
    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"], logger=logging.getLogger("solve"))
    start_run(run_dir, settings, {"name": "nlf", "tag": tag(condition, rule), "fusion": rule,
                                  "condition": condition, "perturbation": list(GRID[condition]),
                                  "participant": participant, "task": task},
              landmark_map=landmark_map, estimator=recorded_estimator(recording))
    saver = CSVSaver(str(run_dir), markers_header=["Frame_0"] + names,
                     joint_angles_header=list(settings.joint_angles_names))
    failed, q_last = 0, None
    for frame, p3d in frames:
        mks = dict(zip(names, marker_filter(p3d)))
        try:
            q = solver.solve(mks)
            ok = np.all(np.isfinite(q))
        except Exception:
            q, ok = None, False
        if not ok:
            failed += 1
            if q_last is None:
                continue
            q = q_last
        q_last = q
        row = OrderedDict([("Frame_0", frame)])
        for name in names:
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = map(float, mks[name])
        saver.save_markers(row)
        saver.save_joint_angles(OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q))))
    saver.close()
    update_run_info(run_dir, solver_failures=failed)


def score(job):
    runs_root, _, dataset, participant, task, condition, rule = job
    from rtcosmik.config_loader import settings
    from rtcosmik.paper.scoring import score_run
    row = score_run(runs_root / participant / task / tag(condition, rule), dataset, participant, task, settings)
    row.update({"condition": condition, "rule": rule, "participant": participant, "task": task})
    return row


def run(args, phase, people, conditions):
    jobs = [(args.output_dir, args.views, args.dataset, p, t, c, r)
            for c in conditions for r in FUSION_RULES for p in people for t in TASKS
            if (args.views / p / f"{t}.npz").exists()]
    print(f"{phase}: {len(jobs)} replays", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for done, _ in enumerate(pool.map(replay, jobs), 1):
            if done % 50 == 0:
                print(f"  {phase}: replayed {done}/{len(jobs)}", flush=True)
        rows = [dict(row, phase=phase) for row in pool.map(score, jobs)]
    return rows


def causal(rows, condition, rule, tasks=None):
    """(participant, task) -> causal error of one condition and rule."""
    return {(r["participant"], r["task"]): float(r["causal_rmse_mean"]) for r in rows
            if r["condition"] == condition and r["rule"] == rule and (tasks is None or r["task"] in tasks)}


def participant_means(values):
    people = {}
    for (participant, _), value in values.items():
        people.setdefault(participant, []).append(value)
    return np.array([np.mean(v) for v in people.values()])


def chosen_levels(rows):
    """Per perturbation, the mildest level that costs the baseline ``MATERIAL_DEG`` on ``rows``."""
    clean = participant_means(causal(rows, "clean", BASELINE)).mean()
    chosen = {}
    for condition, (kind, _) in GRID.items():
        if kind is None or kind in chosen and chosen[kind][1]:
            continue
        cost = participant_means(causal(rows, condition, BASELINE)).mean() - clean
        chosen[kind] = (condition, cost >= MATERIAL_DEG)
    return [condition for condition, _ in chosen.values()]


def table(rows, conditions):
    """Markdown: causal error per condition and rule, and each rule against the baseline."""
    scopes = (("all tasks", None), ("robot tasks", ROBOT_TASKS),
              ("other tasks", tuple(t for t in TASKS if t not in ROBOT_TASKS)))
    lines = []
    for label, tasks in scopes:
        lines += ["", f"**{label}**", "", "| Condition | " + " | ".join(FUSION_RULES) + " |",
                  "|---|" + "---|" * len(FUSION_RULES)]
        for condition in conditions:
            base = causal(rows, condition, BASELINE, tasks)
            cells = []
            for rule in FUSION_RULES:
                mine = causal(rows, condition, rule, tasks)
                means = participant_means(mine)
                cell = f"{means.mean():.2f} ± {means.std(ddof=1):.2f}"
                if rule != BASELINE:
                    diff, _ = st.matched_differences(mine, base)
                    cell += f" ({diff.mean():+.2f}, p = {st.paired_difference(diff)['p']:.3f})"
                cells.append(cell)
            lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    return lines


def effects(rows, conditions):
    """Markdown: what each perturbation did to the recordings' runs (all tasks, baseline rule)."""
    lines = ["", "| Condition | Dropped frames per trial | Solver failures | Marker error (mm) |", "|---|---|---|---|"]
    for condition in conditions:
        mine = [r for r in rows if r["condition"] == condition and r["rule"] == BASELINE]
        lines.append(f"| {condition} | {np.mean([float(r['dropped_frames']) for r in mine]):.1f} | "
                     f"{sum(int(r['solver_failures']) for r in mine)} | "
                     f"{np.mean([float(r['causal_marker_mm']) for r in mine]):.1f} |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/root/workspace/COMFI"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from rtcosmik.provenance import require_clean
    require_clean()
    logging.basicConfig(level=logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)

    def save(rows, text):
        with open(args.out / "trials.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (args.out / "summary.md").write_text("\n".join(text) + "\n")

    validation = run(args, "validation", VALIDATION, list(GRID))
    text = [f"Causal error (deg), mean ± SD over participants; in brackets the mean paired difference to "
            f"`{BASELINE}` and its signed-rank p.", "",
            f"## Validation participants ({len(VALIDATION)}): {' '.join(VALIDATION)}"]
    text += table(validation, list(GRID)) + effects(validation, list(GRID))
    reported = ["clean"] + chosen_levels(validation)
    text += ["", f"Levels kept for the other participants: {', '.join(reported[1:])} "
             f"(mildest costing `{BASELINE}` at least {MATERIAL_DEG} deg, else the strongest)."]
    save(validation, text)

    held_out = run(args, "held_out", HELD_OUT, reported)
    text += ["", f"## Held-out participants ({len(HELD_OUT)}): {' '.join(HELD_OUT)}"]
    text += table(held_out, reported) + effects(held_out, reported)
    save(validation + held_out, text)
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
