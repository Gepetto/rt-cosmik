#!/usr/bin/env python3
"""Fit NLF's marker vertices with the method that produced FastSAM's marker map.

FastSAM's markers were learned by ``learn_cosmik_mhr_markers_multisubject.py``:
for each marker, candidate mesh vertices near an anatomical keypoint are scored
against the COMFI mocap marker after a per-frame similarity alignment, the
scores are aggregated by median over frames, then tasks, then subjects, and a
unique vertex per marker is picked by linear assignment. The map is validated on
held-out subjects, then refit on all 17.

NLF's markers were picked by hand. Running the same procedure for NLF makes the
two modalities' marker definitions come from the same process, so the modality
comparison is not also a comparison of how markers were chosen.

Everything model-agnostic is imported from that script rather than copied: the
robust Umeyama alignment, the anatomical anchors, the search radii, the mocap
loader, the balanced scoring and the assignment rule. What changes is only what
has to change:

* the mesh: NLF's 10475 SMPL-X canonical vertices instead of MHR's 18439;
* the keypoints the anchors come from. MHR has 70 named keypoints; NLF has
  SMPL-X, so each MHR keypoint the method uses is taken from its SMPL-X
  equivalent (:data:`MHR70_FROM_SMPLX`);
* the marker list: the 29 body markers of the parity set. The two posterior
  thoracic markers the FastSAM map also learned are dropped by parity anyway.

The pelvis markers are fitted but not used: they keep their hand-picked vertices
(:data:`KEEP_HAND_PICKED`). Fitted one by one, the posterior markers each land
closer to mocap, but together they sit too close and too high: the pelvis frame
the IK builds from them tilts 8.3 deg further than mocap's (hand-picked: 2.5),
and with the thoracic joints locked the thorax follows it. Replayed on the same
NLF output over 8 trials, keeping the hand-picked pelvis lowered whole-body RMSE
by 0.8 deg with one camera and 3.1 deg with 2D triangulation, cut shoulder flips
in SideOverhead, and cost nothing with four cameras (12.05 -> 12.13 deg).
Constraining the PSIS pair to mocap's separation and tilt inside the fit only
half corrected the tilt and did slightly worse, so it was not kept.

The medial ankle markers keep their hand-picked vertices for the same reason.
Candidate vertices there tie on marker error within a millimetre, so the fitted
one changes with the participants fitted on, while the ankle frame the IK builds
from the two malleoli does not tolerate the change: left ankle
inversion/eversion differs by about 28 deg between the two candidates.

The six facial markers are not fitted, for either modality: mocap has no facial
landmarks to fit them against.

Frames come from ``sample_nlf_dense.py``, which ran NLF on camera 0 at exactly
the candidate frames the FastSAM fit drew from.

One map is fitted per fold (``rtcosmik.folds``), on the other fifteen
participants, and scored on the fold's own three.

    python3 scripts/python/paper/fit_nlf_marker_map.py \\
        --samples /root/workspace/nlf_dense_samples --out-dir config/landmark_maps
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

FASTSAM_DIR = Path("/root/workspace/COMFI/fastsam/results_multicam")
TASKS = ("Screwing", "SideOverhead", "RobotWelding", "RobotPolishing", "Polishing", "Lifting")
FASTSAM_LEARNER = FASTSAM_DIR / "learn_cosmik_mhr_markers_multisubject.py"
VERTEX_COUNT = 10475
FRAMES_PER_SEQUENCE = 20
TOP_CANDIDATES = 256
#: Markers whose fitted vertex is recorded but not used (see module docstring).
KEEP_HAND_PICKED = ("RASI", "LASI", "RPSI", "LPSI", "RMANK", "LMANK")

#: NLF's hand-picked SMPL-X vertices for the markers used as surface anchors.
HAND_PICKED = {
    "RASI": 8421, "LASI": 5727, "RPSI": 8371, "LPSI": 5677, "C7": 5484,
    "RSHO": 6629, "LSHO": 3878, "RELB": 7040, "LELB": 4302, "RMELB": 7105,
    "LMELB": 4369, "RWRI": 7584, "LWRI": 4848, "RMWRI": 7457, "LMWRI": 4721,
    "RKNE": 6401, "LKNE": 3640, "RMKNE": 6407, "LMKNE": 3646, "RANK": 8576,
    "LANK": 5882, "RMANK": 8680, "LMANK": 8892, "R5MHD": 8474, "L5MHD": 5780,
    "RTOE": 8463, "LTOE": 5770, "RHEE": 8635, "LHEE": 8846,
}

#: Each MHR70 keypoint the fitting method reads, and where it comes from on SMPL-X.
#: ("joint", i) is SMPL-X joint i from the joint regressor; ("vertex", i) is a
#: canonical vertex.
#:
#: * Toes and heels: SMPL-X has no such joints. The vertices are the standard
#:   SMPL-X toe and heel keypoint vertices, which are also NLF's hand-picked foot
#:   markers.
#: * Acromion: the SMPL-X shoulder joint is the gleno-humeral centre, several
#:   centimetres inside the acromion MHR's keypoint marks, so the hand-picked
#:   shoulder surface vertex is used instead.
#: * Olecranon and cubital fossa: no SMPL-X counterpart. The method only uses them
#:   to nudge the elbow search centre inside an 18 cm radius, so the elbow joint
#:   stands in for both.
MHR70_FROM_SMPLX = {
    7: ("joint", 18), 8: ("joint", 19),            # elbows
    9: ("joint", 1), 10: ("joint", 2),             # hips
    11: ("joint", 4), 12: ("joint", 5),            # knees
    13: ("joint", 7), 14: ("joint", 8),            # ankles
    15: ("vertex", 5770), 16: ("vertex", 5780), 17: ("vertex", 8846),   # left toes, heel
    18: ("vertex", 8463), 19: ("vertex", 8474), 20: ("vertex", 8635),   # right toes, heel
    41: ("joint", 21), 62: ("joint", 20),          # wrists
    63: ("joint", 18), 64: ("joint", 19),          # olecranon -> elbow joint
    65: ("joint", 18), 66: ("joint", 19),          # cubital fossa -> elbow joint
    67: ("vertex", 3878), 68: ("vertex", 6629),    # acromion -> shoulder surface
    69: ("joint", 12),                             # neck
}


def load_learner():
    spec = importlib.util.spec_from_file_location("fastsam_learner", FASTSAM_LEARNER)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Parity uses the 29 body markers; drop the two thoracic markers everywhere
    # the module reads its marker list (mocap loading, anchors, ordering).
    module.MARKERS = module.BASE_MARKERS
    return module


def keypoints70(vertices, regressor):
    """An MHR70-shaped keypoint array built from one frame of NLF vertices."""
    joints = regressor @ vertices
    out = np.full((70, 3), np.nan)
    for index, (kind, source) in MHR70_FROM_SMPLX.items():
        out[index] = joints[source] if kind == "joint" else vertices[source]
    return out


@dataclass
class Sample:
    participant: str
    subject: str
    task: str
    mocap: np.ndarray
    frames: np.ndarray
    vertices: np.ndarray       # (S, V, 3) for the kept frames
    keypoints: np.ndarray      # (S, 70, 3)
    scales: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    anchor_rmse_m: np.ndarray


def sample_trial(learner, regressor, npz_path, mocap_path, participant, subject, task,
                 requested):
    """Port of ``sample_sequence``: same candidates, same acceptance, same thinning."""
    data = np.load(npz_path)
    mocap = learner.load_aligned_mocap(mocap_path)
    if len(mocap) != int(data["n_frames"]):
        raise ValueError(f"{participant}/{task}: mocap {len(mocap)} != video {data['n_frames']}")

    kept = defaultdict(list)
    for slot, frame in enumerate(data["frames"]):
        if not data["valid"][slot]:
            continue
        vertices = data["vertices"][slot].astype(np.float64)
        kp = keypoints70(vertices, regressor)
        source, target = learner.alignment_anchors(kp, mocap[frame])
        try:
            scale, rotation, translation = learner.similarity(source, target)
        except (ValueError, np.linalg.LinAlgError):
            continue
        fitted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(fitted - target, axis=1)
        finite = residual[np.isfinite(residual)]
        if len(finite) < 6:
            continue
        kept["frames"].append(int(frame))
        kept["vertices"].append(vertices.astype(np.float32))
        kept["keypoints"].append(kp)
        kept["scales"].append(scale)
        kept["rotations"].append(rotation)
        kept["translations"].append(translation)
        kept["rmse"].append(float(np.sqrt(np.mean(finite * finite))))

    if len(kept["frames"]) < min(10, requested):
        raise ValueError(f"{participant}/{task}: only {len(kept['frames'])} usable frames")
    if len(kept["frames"]) > requested:
        pick = np.linspace(0, len(kept["frames"]) - 1, requested).round().astype(np.int64)
        kept = {key: [values[i] for i in pick] for key, values in kept.items()}
    return Sample(participant, subject, task, mocap,
                  np.asarray(kept["frames"]), np.asarray(kept["vertices"]),
                  np.asarray(kept["keypoints"]), np.asarray(kept["scales"]),
                  np.asarray(kept["rotations"]), np.asarray(kept["translations"]),
                  np.asarray(kept["rmse"]))


def balanced_scores(learner, samples, candidates, marker_index):
    """Port of ``balanced_candidate_scores``: median over frames, tasks, subjects."""
    by_subject = defaultdict(list)
    for sample in samples:
        errors = []
        for slot, frame in enumerate(sample.frames):
            target = sample.mocap[frame, marker_index]
            if not np.isfinite(target).all():
                continue
            points = sample.vertices[slot, candidates].astype(np.float64)
            aligned = (sample.scales[slot] * (points @ sample.rotations[slot].T)
                       + sample.translations[slot])
            errors.append(np.linalg.norm(aligned - target, axis=1))
        if errors:
            by_subject[sample.subject].append(np.median(errors, axis=0))
    if not by_subject:
        raise RuntimeError(f"no observations for marker {learner.MARKERS[marker_index][0]}")
    return np.median([np.median(tasks, axis=0) for tasks in by_subject.values()], axis=0)


def choose_vertices(learner, samples, top_candidates):
    """Port of ``choose_vertices`` for SMPL-X vertices."""
    reference = min(samples, key=lambda s: float(np.median(s.anchor_rmse_m)))
    slot = int(np.argmin(reference.anchor_rmse_m))
    ref_vertices = reference.vertices[slot].astype(np.float64)
    ref_keypoints = reference.keypoints[slot]

    rankings = []
    for marker_index, (label, _) in enumerate(learner.MARKERS):
        anchor = learner.marker_anchor(ref_keypoints, label)
        radius = learner.SEARCH_RADIUS[learner.marker_region(label)]
        candidates = np.flatnonzero(np.linalg.norm(ref_vertices - anchor, axis=1) <= radius)
        if len(candidates) < 10:
            raise RuntimeError(f"only {len(candidates)} SMPL-X candidates for {label}")
        scores = balanced_scores(learner, samples, candidates, marker_index)
        order = np.argsort(scores)[: min(top_candidates, len(scores))]
        rankings.append((candidates[order], scores[order], len(candidates)))
        print(f"{marker_index + 1:2d}/{len(learner.MARKERS)} {label:6s}: best="
              f"{int(candidates[order[0]]):5d}, balanced median={scores[order[0]] * 1000:5.1f} mm",
              flush=True)

    union = np.unique(np.concatenate([ids for ids, _, _ in rankings]))
    lookup = {int(v): c for c, v in enumerate(union)}
    cost = np.full((len(learner.MARKERS), len(union)), 1e3)
    for marker_index, (ids, scores, _) in enumerate(rankings):
        for vertex, score in zip(ids, scores):
            cost[marker_index, lookup[int(vertex)]] = float(score)
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(learner.MARKERS))) or np.any(cost[rows, columns] >= 1e3):
        raise RuntimeError("no unique assignment for all markers")
    selected = union[columns].astype(np.int64)
    selection = {
        label: {"vertex_index": HAND_PICKED[label] if label in KEEP_HAND_PICKED else int(vertex),
                "fitted_vertex_index": int(vertex),
                "hand_picked_vertex_index": HAND_PICKED[label],
                "candidate_count": int(rankings[i][2]),
                "balanced_error_mm": float(cost[i, columns[i]] * 1000)}
        for i, ((label, _), vertex) in enumerate(zip(learner.MARKERS, selected))}
    return selected, selection, {"participant": reference.participant,
                                 "task": reference.task,
                                 "frame": int(reference.frames[slot])}


def evaluate(learner, samples, selected):
    """Port of ``evaluate_vertices``: aligned marker error, overall and per marker."""
    per_marker = [[] for _ in learner.MARKERS]
    for sample in samples:
        for slot, frame in enumerate(sample.frames):
            points = sample.vertices[slot, selected].astype(np.float64)
            aligned = (sample.scales[slot] * (points @ sample.rotations[slot].T)
                       + sample.translations[slot])
            residual = np.linalg.norm(aligned - sample.mocap[frame], axis=1)
            for i, value in enumerate(residual):
                if np.isfinite(value):
                    per_marker[i].append(float(value))
    overall = np.asarray([v for values in per_marker for v in values])
    return {
        "sampled_sequence_count": len(samples),
        "marker_observation_count": int(len(overall)),
        "overall_median_mm": float(np.median(overall) * 1000),
        "per_marker_median_mm": {label: float(np.median(v) * 1000)
                                 for (label, _), v in zip(learner.MARKERS, per_marker)},
    }


def fit_fold(learner, samples, fold, hand):
    """Fold ``fold``'s map: fitted on its fifteen fitting participants, scored on its three."""
    from rtcosmik.folds import FOLDS, fitting_participants
    training = [s for s in samples if s.participant in fitting_participants(fold)]
    held_out = [s for s in samples if s.participant in FOLDS[fold]]
    print(f"\nfold {fold}: held out {', '.join(FOLDS[fold])}")
    fitted, selection, reference = choose_vertices(learner, training, TOP_CANDIDATES)
    used = np.asarray([selection[label]["vertex_index"] for label, _ in learner.MARKERS])
    return {
        "front_end": "nlf",
        "fold": fold,
        "fitted_on": list(fitting_participants(fold)),
        "vertices": {label: entry["vertex_index"] for label, entry in selection.items()},
        "model": "NLF, SMPL-X canonical vertices",
        "method": "balanced_multisubject_mocap_fit, ported from "
                  "learn_cosmik_mhr_markers_multisubject.py",
        "mhr70_from_smplx": {str(k): list(v) for k, v in MHR70_FROM_SMPLX.items()},
        "camera": 0,
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "top_candidates": TOP_CANDIDATES,
        "reference": reference,
        "markers": selection,
        "evaluation": {
            "used_on_fitted": evaluate(learner, training, used),
            "used_on_held_out": evaluate(learner, held_out, used),
            "fitted_on_held_out": evaluate(learner, held_out, fitted),
            "hand_picked_on_held_out": evaluate(learner, held_out, hand),
        },
        "not_fitted": ["Nose", "Head", "REar", "LEar", "REye", "LEye"],
        "kept_hand_picked": list(KEEP_HAND_PICKED),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/root/workspace/COMFI"))
    ap.add_argument("--body-models", type=Path,
                    default=REPO / "weights" / "body_models" / "smplx")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    from rtcosmik.folds import FOLDS, PARTICIPANTS
    learner = load_learner()
    regressor = np.load(args.body_models / "SMPLX_NEUTRAL.npz", allow_pickle=True)["J_regressor"]
    if regressor.shape != (55, VERTEX_COUNT):
        raise ValueError(f"unexpected SMPL-X joint regressor {regressor.shape}")

    samples = []
    for participant in PARTICIPANTS:
        for task in TASKS:
            npz = args.samples / participant / f"{task}.npz"
            mocap = args.dataset / "mocap" / "aligned" / participant / task / "markers_trajectories.csv"
            samples.append(sample_trial(learner, regressor, npz, mocap, participant,
                                        participant, task, FRAMES_PER_SEQUENCE))
    print(f"sampled {len(samples)} trials from {len({s.participant for s in samples})} participants")
    hand = np.asarray([HAND_PICKED[label] for label, _ in learner.MARKERS])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(len(FOLDS)):
        payload = fit_fold(learner, samples, fold, hand)
        path = args.out_dir / f"nlf_fold{fold}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        evaluation = payload["evaluation"]
        rows.append((fold, evaluation["used_on_held_out"]["overall_median_mm"],
                     evaluation["hand_picked_on_held_out"]["overall_median_mm"]))
        print(f"written to {path}")
    print(f"\nheld-out median marker error (mm)\n{'fold':<6}{'fitted (used)':>15}{'hand-picked':>13}")
    for fold, used, hand_mm in rows:
        print(f"{fold:<6}{used:>15.1f}{hand_mm:>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
