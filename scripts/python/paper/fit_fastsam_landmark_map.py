#!/usr/bin/env python3
"""Fit FastSAM-3D's landmark map once per fold, from the stored camera-2 meshes.

The method is ``learn_cosmik_mhr_markers_multisubject.py``, imported and run
unchanged on each fold's fitting participants. Output names participants by ID.

    python3 scripts/python/paper/fit_fastsam_landmark_map.py \\
        --results /root/workspace/COMFI/fastsam/results_multicam --out-dir config/landmark_maps
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

CAMERA = 2
FRAMES_PER_SEQUENCE = 20
TOP_CANDIDATES = 256
TASK_COUNT = 6


def load_learner(results):
    path = results / "learn_cosmik_mhr_markers_multisubject.py"
    spec = importlib.util.spec_from_file_location("fastsam_learner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sequences(learner, results, reference_root):
    """SequenceInfo per (participant, task) from the camera-2 meshes, and the rejected trials.

    As in the learner, a trial whose mesh and reference lengths differ is rejected,
    and so is every trial of a participant left without all six tasks.
    """
    # The learner's excluded folder is 3361, whose camera-2 meshes come from the common script.
    ids = {**learner.SUBJECT_IDS, **dict.fromkeys(learner.EXCLUDED_SUBJECTS, "3361")}
    found, rejected = [], []
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        pid = next((ids[c] for c in (folder.name, folder.name.rstrip("_"), folder.name + "_")
                    if c in ids), None)
        if pid is None:
            continue
        for raw_task, task in learner.TASKS.items():
            result_dir = folder / raw_task / f"camera_{CAMERA}" / "calibrated"
            frames, error = learner.readable_result(result_dir)
            if error:
                raise RuntimeError(f"{pid}/{task}: {error}")
            reference_csv = reference_root / pid / task / "markers_trajectories.csv"
            reference_frames = len(learner.load_aligned_mocap(reference_csv))
            if reference_frames != frames:
                rejected.append({"participant": pid, "task": task, "reason":
                                 f"length mismatch: mesh={frames}, reference={reference_frames}"})
                continue
            found.append(learner.SequenceInfo(subject=pid, subject_id=pid, task=task,
                                              result_dir=result_dir, mocap_csv=reference_csv,
                                              frame_count=frames))
    incomplete = {p for p in {i.subject_id for i in found}
                  if sum(i.subject_id == p for i in found) != TASK_COUNT}
    rejected += [{"participant": i.subject_id, "task": i.task, "reason": "participant incomplete"}
                 for i in found if i.subject_id in incomplete]
    return [i for i in found if i.subject_id not in incomplete], rejected


def fit_fold(learner, sampled, fold, template, rejected):
    """Fold ``fold``'s map, in the exporter's schema plus the fold fields."""
    from rtcosmik.folds import FOLDS, fitting_participants
    training = [s for s in sampled if s.info.subject_id in fitting_participants(fold)]
    fitted_on = sorted({s.info.subject_id for s in training})
    held_out = [s for s in sampled if s.info.subject_id in FOLDS[fold]]
    print(f"\nfold {fold}: held out {', '.join(FOLDS[fold])}", flush=True)
    selected, selection, reference = learner.choose_vertices(training, TOP_CANDIDATES)
    _, fitted_metrics = learner.evaluate_vertices(training, selected)
    held_marker, held_metrics = learner.evaluate_vertices(held_out, selected)
    markers = {}
    for (label, _), entry, metric in zip(learner.MARKERS, selection, held_marker):
        markers[label] = dict(entry, held_out=metric)
    return {
        "front_end": "fastsam",
        "fold": fold,
        "fitted_on": fitted_on,
        "vertices": {label: entry["vertex_index"] for label, entry in markers.items()},
        "index_base": 0,
        "vertex_count": template["vertex_count"],
        "method": template["method"],
        "layout": template["layout"],
        "results_camera_mode": "calibrated",
        "camera": CAMERA,
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "top_candidates": TOP_CANDIDATES,
        "reference": reference,
        "markers": markers,
        "face_keypoints": template["face_keypoints"],
        "evaluation": {"fitted": fitted_metrics, "held_out": held_metrics},
        "rejected_sequences": rejected,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/root/workspace/COMFI"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    from rtcosmik.folds import FOLDS, PARTICIPANTS
    learner = load_learner(args.results)
    template = json.loads((args.results / "cosmik_mhr_marker_map_17subjects_tv8_tv12.json")
                          .read_text())
    infos, rejected = sequences(learner, args.results, args.dataset / "mocap" / "aligned")
    if len(infos) + len(rejected) != TASK_COUNT * len(PARTICIPANTS):
        raise RuntimeError(f"incomplete meshes: {len(infos)} trials, {len(rejected)} rejected")
    for item in rejected:
        print(f"rejected {item['participant']}/{item['task']}: {item['reason']}")
    sampled = []
    for index, info in enumerate(infos, 1):
        sampled.append(learner.sample_sequence(info, FRAMES_PER_SEQUENCE))
        print(f"sampled {index:3d}/{len(infos)} {info.subject_id}/{info.task}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(len(FOLDS)):
        payload = fit_fold(learner, sampled, fold, template, rejected)
        path = args.out_dir / f"fastsam_fold{fold}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        rows.append((fold, payload["evaluation"]["fitted"]["overall_median_mm"],
                     payload["evaluation"]["held_out"]["overall_median_mm"]))
        print(f"written to {path}", flush=True)
    print(f"\nmedian marker error (mm)\n{'fold':<6}{'fitted':>8}{'held-out':>10}")
    for fold, fitted, held in rows:
        print(f"{fold:<6}{fitted:>8.1f}{held:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
