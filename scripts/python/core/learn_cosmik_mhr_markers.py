#!/usr/bin/env python3
"""Learn fixed MHR surface vertices for the 29-marker RT-COSMIK set.

MHR meshes are first transformed from the camera frame to the mocap/world
frame with the COMFI extrinsics.  One robust global residual similarity is
then learned on the fit frames to compensate for systematic MHR scale and
pose biases.  No frame-wise alignment is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


MARKERS = (
    ("RASI", "r.ASIS_study"), ("LASI", "L.ASIS_study"),
    ("RPSI", "r.PSIS_study"), ("LPSI", "L.PSIS_study"),
    ("C7", "C7_study"),
    ("RSHO", "r_shoulder_study"), ("LSHO", "L_shoulder_study"),
    ("RELB", "r_lelbow_study"), ("LELB", "L_lelbow_study"),
    ("RMELB", "r_melbow_study"), ("LMELB", "L_melbow_study"),
    ("RWRI", "r_lwrist_study"), ("LWRI", "L_lwrist_study"),
    ("RMWRI", "r_mwrist_study"), ("LMWRI", "L_mwrist_study"),
    ("RKNE", "r_knee_study"), ("LKNE", "L_knee_study"),
    ("RMKNE", "r_mknee_study"), ("LMKNE", "L_mknee_study"),
    ("RANK", "r_ankle_study"), ("LANK", "L_ankle_study"),
    ("RMANK", "r_mankle_study"), ("LMANK", "L_mankle_study"),
    ("R5MHD", "r_5meta_study"), ("L5MHD", "L_5meta_study"),
    ("RTOE", "r_toe_study"), ("LTOE", "L_toe_study"),
    ("RHEE", "r_calc_study"), ("LHEE", "L_calc_study"),
)

FACE_KEYPOINTS = (
    ("Nose", 0), ("REar", 4), ("LEar", 3), ("REye", 2), ("LEye", 1),
)

# MHR70 anchors constrain each search to the correct anatomical neighbourhood.
MARKER_ANCHORS = {
    "RASI": (10,), "RPSI": (10,), "LASI": (9,), "LPSI": (9,),
    "C7": (69,), "RSHO": (68,), "LSHO": (67,),
    "RELB": (8, 64), "RMELB": (8, 66),
    "LELB": (7, 63), "LMELB": (7, 65),
    "RWRI": (41,), "RMWRI": (41,), "LWRI": (62,), "LMWRI": (62,),
    "RKNE": (12,), "RMKNE": (12,), "LKNE": (11,), "LMKNE": (11,),
    "RANK": (14,), "RMANK": (14,), "LANK": (13,), "LMANK": (13,),
    "R5MHD": (18, 19), "RTOE": (18, 19), "RHEE": (20,),
    "L5MHD": (15, 16), "LTOE": (15, 16), "LHEE": (17,),
}

SEARCH_RADIUS = {
    "pelvis": 0.28, "torso": 0.25, "shoulder": 0.20,
    "elbow": 0.18, "wrist": 0.16, "knee": 0.20,
    "ankle": 0.18, "foot": 0.24,
}


def marker_region(label: str) -> str:
    if label in {"RASI", "LASI", "RPSI", "LPSI"}: return "pelvis"
    if label == "C7": return "torso"
    if label.endswith("SHO"): return "shoulder"
    if "ELB" in label: return "elbow"
    if "WRI" in label: return "wrist"
    if "KNE" in label: return "knee"
    if "ANK" in label: return "ankle"
    return "foot"


def load_mocap(path: Path) -> np.ndarray:
    columns = [f"{name}_{axis}" for _, name in MARKERS for axis in "xyz"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(columns) - set(reader.fieldnames or []))
        if missing:
            raise ValueError("Colonnes mocap absentes: " + ", ".join(missing))
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[name]) for name in columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Valeur mocap invalide ligne {line}") from exc
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(MARKERS), 3) * 0.001


def load_extrinsics(
    path: Path,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Load a COMFI frame_from -> frame_to similarity transform."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    extrinsics = payload.get("camera_extrinsics", payload)
    required = {"rotation_matrix", "translation_vector"}
    missing = sorted(required - set(extrinsics))
    if missing:
        raise ValueError("Champs extrinseques absents: " + ", ".join(missing))

    rotation = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsics["translation_vector"], dtype=np.float64)
    scale = float(extrinsics.get("scale_factor", 1.0))
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"Extrinseques invalides: rotation={rotation.shape}, translation={translation.shape}"
        )
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.isfinite(scale):
        raise ValueError("Valeur extrinseque non finie")
    if scale <= 0:
        raise ValueError(f"scale_factor extrinseque invalide: {scale}")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("La matrice de rotation extrinseque n'est pas orthonormale")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("La rotation extrinseque n'est pas une rotation propre")
    metadata = {
        "frame_from": str(extrinsics.get("frame_from", "camera")),
        "frame_to": str(extrinsics.get("frame_to", "world")),
        "rms_error": extrinsics.get("rms_error"),
        "source_file": extrinsics.get("source_file"),
    }
    return scale, rotation, translation, metadata


def transform_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply p_to = scale * R @ p_from + translation to (..., 3) points."""
    return scale * (np.asarray(points) @ rotation.T) + translation


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Proper Umeyama similarity, robustly refitted after removing outliers."""
    keep = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    if keep.sum() < 6:
        raise ValueError("Pas assez d'ancres valides pour l'alignement")
    for _ in range(3):
        x, y = source[keep], target[keep]
        cx, cy = x.mean(0), y.mean(0)
        xc, yc = x - cx, y - cy
        u, singular, vt = np.linalg.svd(xc.T @ yc)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        scale = float(singular.sum() / np.sum(xc * xc))
        translation = cy - scale * (rotation @ cx)
        residual = np.linalg.norm(
            scale * (source @ rotation.T) + translation - target, axis=1
        )
        finite_residual = residual[np.isfinite(residual)]
        threshold = max(0.035, float(np.percentile(finite_residual, 85)))
        new_keep = np.isfinite(residual) & (residual <= threshold)
        if new_keep.sum() < 6 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return scale, rotation, translation


def alignment_anchors(keypoints: np.ndarray, mocap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    marker_id = {label: i for i, (label, _) in enumerate(MARKERS)}
    specs = (
        ((68,), ("RSHO",)), ((67,), ("LSHO",)),
        ((8,), ("RELB", "RMELB")), ((7,), ("LELB", "LMELB")),
        ((41,), ("RWRI", "RMWRI")), ((62,), ("LWRI", "LMWRI")),
        ((10,), ("RASI", "RPSI")), ((9,), ("LASI", "LPSI")),
        ((12,), ("RKNE", "RMKNE")), ((11,), ("LKNE", "LMKNE")),
        ((14,), ("RANK", "RMANK")), ((13,), ("LANK", "LMANK")),
        ((20,), ("RHEE",)), ((17,), ("LHEE",)), ((69,), ("C7",)),
    )
    source, target = [], []
    for keypoint_ids, marker_names in specs:
        source.append(np.mean(keypoints[list(keypoint_ids)], axis=0))
        target.append(np.mean(mocap[[marker_id[name] for name in marker_names]], axis=0))
    return np.asarray(source), np.asarray(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("camera_0"))
    parser.add_argument("--mocap-csv", type=Path)
    parser.add_argument("--extrinsics-yaml", type=Path, required=True)
    parser.add_argument("--output-map", type=Path)
    parser.add_argument(
        "--output-npy", type=Path,
        help="Trajectoires apres extrinseques et correction residuelle globale",
    )
    parser.add_argument(
        "--output-world-npy", type=Path,
        help="Trajectoires apres extrinseques COMFI uniquement",
    )
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--top-candidates", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.input_dir.resolve()
    mocap_path = args.mocap_csv or root / "mocap_downsampled_to_40hz.csv"
    map_path = args.output_map or root / "cosmik_mhr_marker_map_auto.json"
    npy_path = args.output_npy or root / "cosmik_markers_mocap_aligned.npy"
    world_npy_path = args.output_world_npy or root / "cosmik_markers_world.npy"
    output_paths = (map_path, npy_path, world_npy_path)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Les trois chemins de sortie doivent etre differents")
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Le fichier existe deja: {path} (utiliser --overwrite)")

    vertices = np.load(root / "vertices_cam.npy", mmap_mode="r")
    keypoints = np.load(root / "keypoints70_cam.npy", mmap_mode="r")
    mocap = load_mocap(mocap_path)
    calibration_scale, calibration_rotation, calibration_translation, calibration_meta = (
        load_extrinsics(args.extrinsics_yaml)
    )
    if vertices.ndim != 3 or vertices.shape[1:] != (18439, 3):
        raise ValueError(f"vertices_cam.npy invalide: {vertices.shape}")
    if keypoints.shape != (len(vertices), 70, 3) or len(mocap) != len(vertices):
        raise ValueError(
            f"Longueurs/formes incompatibles: {vertices.shape}, {keypoints.shape}, {mocap.shape}"
        )
    valid_path = root / "valid.npy"
    valid = np.load(valid_path).astype(bool) if valid_path.exists() else np.ones(len(vertices), bool)
    valid &= np.isfinite(keypoints).all(axis=(1, 2)) & np.isfinite(mocap).all(axis=(1, 2))
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) < 10:
        raise ValueError("Moins de 10 frames communes valides")
    count = min(max(10, args.sample_count), len(valid_ids))
    sample_ids = valid_ids[np.linspace(0, len(valid_ids) - 1, count).round().astype(int)]
    fit_slots = np.arange(0, len(sample_ids), 2, dtype=np.int64)
    validation_slots = np.arange(1, len(sample_ids), 2, dtype=np.int64)
    if not len(validation_slots):
        validation_slots = fit_slots

    calibrated_anchors = []
    target_anchors = []
    for frame in sample_ids:
        source, target = alignment_anchors(keypoints[frame], mocap[frame])
        calibrated_anchors.append(transform_points(
            source, calibration_scale, calibration_rotation, calibration_translation
        ))
        target_anchors.append(target)
    calibrated_anchors = np.asarray(calibrated_anchors)
    target_anchors = np.asarray(target_anchors)

    residual_scale, residual_rotation, residual_translation = similarity(
        calibrated_anchors[fit_slots].reshape(-1, 3),
        target_anchors[fit_slots].reshape(-1, 3),
    )
    corrected_anchors = transform_points(
        calibrated_anchors, residual_scale, residual_rotation, residual_translation
    )
    anchor_errors = np.linalg.norm(corrected_anchors - target_anchors, axis=2)
    anchor_rmse = np.sqrt(np.mean(anchor_errors * anchor_errors, axis=1))
    reference_slot = int(fit_slots[np.argmin(anchor_rmse[fit_slots])])
    reference_frame = int(sample_ids[reference_slot])
    reference_vertices = transform_points(
        transform_points(
            vertices[reference_frame],
            calibration_scale, calibration_rotation, calibration_translation,
        ),
        residual_scale, residual_rotation, residual_translation,
    )
    reference_keypoints = transform_points(
        transform_points(
            keypoints[reference_frame],
            calibration_scale, calibration_rotation, calibration_translation,
        ),
        residual_scale, residual_rotation, residual_translation,
    )

    rankings: list[tuple[np.ndarray, np.ndarray, int]] = []
    all_scores: list[dict[int, float]] = []
    for marker_index, (label, _) in enumerate(MARKERS):
        anchor_ids = MARKER_ANCHORS[label]
        anchor = reference_keypoints[list(anchor_ids)].mean(axis=0)
        radius = SEARCH_RADIUS[marker_region(label)]
        candidate_ids = np.flatnonzero(
            np.linalg.norm(reference_vertices - anchor, axis=1) <= radius
        )
        if len(candidate_ids) < 10:
            raise RuntimeError(f"Seulement {len(candidate_ids)} candidats pour {label}")
        distances = np.empty((len(fit_slots), len(candidate_ids)), dtype=np.float32)
        for output_slot, slot in enumerate(fit_slots):
            frame = sample_ids[slot]
            calibrated = transform_points(
                vertices[frame, candidate_ids],
                calibration_scale, calibration_rotation, calibration_translation,
            )
            aligned = transform_points(
                calibrated, residual_scale, residual_rotation, residual_translation
            )
            distances[output_slot] = np.linalg.norm(
                aligned - mocap[frame, marker_index], axis=1
            )
        scores = np.median(distances, axis=0)
        order = np.argsort(scores)[: min(args.top_candidates, len(scores))]
        ranked_ids, ranked_scores = candidate_ids[order], scores[order]
        rankings.append((ranked_ids, ranked_scores, len(candidate_ids)))
        all_scores.append({int(i): float(s) for i, s in zip(ranked_ids, ranked_scores)})

    # A marker set cannot assign the same surface vertex to two physical markers.
    union = np.unique(np.concatenate([ids for ids, _, _ in rankings]))
    lookup = {int(vertex): column for column, vertex in enumerate(union)}
    cost = np.full((len(MARKERS), len(union)), 1e3, dtype=np.float64)
    for marker, score_by_vertex in enumerate(all_scores):
        for vertex, score in score_by_vertex.items():
            cost[marker, lookup[vertex]] = score
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(MARKERS))) or np.any(cost[rows, columns] >= 1e3):
        raise RuntimeError("Impossible de construire une affectation unique des marqueurs")
    selected = union[columns].astype(np.int64)

    diagnostics = []
    for marker_index, ((label, csv_name), vertex) in enumerate(zip(MARKERS, selected)):
        errors = []
        for slot in validation_slots:
            frame = sample_ids[slot]
            calibrated = transform_points(
                vertices[frame, vertex],
                calibration_scale, calibration_rotation, calibration_translation,
            )
            aligned = transform_points(
                calibrated, residual_scale, residual_rotation, residual_translation
            )
            errors.append(float(np.linalg.norm(aligned - mocap[frame, marker_index])))
        errors = np.asarray(errors)
        candidate_count = rankings[marker_index][2]
        diagnostics.append({
            "number": marker_index + 1,
            "csv_name": csv_name,
            "vertex_index": int(vertex),
            "mhr70_anchor_indices": list(MARKER_ANCHORS[label]),
            "candidate_count": int(candidate_count),
            "median_validation_error_mm": float(np.median(errors) * 1000),
            "mean_validation_error_mm": float(np.mean(errors) * 1000),
            "p95_validation_error_mm": float(np.percentile(errors, 95) * 1000),
        })

    camera_surface_markers = np.asarray(vertices[:, selected, :], dtype=np.float64)
    camera_face_markers = np.asarray(
        keypoints[:, [index for _, index in FACE_KEYPOINTS], :], dtype=np.float64
    )
    camera_markers = np.concatenate((camera_surface_markers, camera_face_markers), axis=1)
    world_markers = transform_points(
        camera_markers, calibration_scale, calibration_rotation, calibration_translation
    )
    output_markers = transform_points(
        world_markers, residual_scale, residual_rotation, residual_translation
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(world_npy_path, world_markers.astype(np.float32))
    np.save(npy_path, output_markers.astype(np.float32))

    def anchor_stats(slots: np.ndarray, errors: np.ndarray) -> dict[str, float]:
        selected_errors = errors[slots]
        frame_rmse = np.sqrt(np.mean(selected_errors * selected_errors, axis=1))
        return {
            "frame_rmse_median_mm": float(np.median(frame_rmse) * 1000),
            "point_error_median_mm": float(np.median(selected_errors) * 1000),
            "point_error_p95_mm": float(np.percentile(selected_errors, 95) * 1000),
        }

    calibrated_anchor_errors = np.linalg.norm(calibrated_anchors - target_anchors, axis=2)
    payload = {
        "index_base": 0,
        "vertex_count": int(vertices.shape[1]),
        "method": "calibrated_global_similarity_mhr70_constrained_multiframe_mocap_fit",
        "anatomically_validated": False,
        "layout": "mocap29",
        "trajectory_layout": [label for label, _ in MARKERS] + [
            label for label, _ in FACE_KEYPOINTS
        ],
        "coordinate_note": (
            "output_world_npy uses the COMFI frame_from->frame_to extrinsics only; "
            "output_npy additionally uses one residual similarity learned on fit frames."
        ),
        "training_input_dir": str(root),
        "training_mocap_csv": str(mocap_path.resolve()),
        "output_world_npy": str(world_npy_path.resolve()),
        "output_npy": str(npy_path.resolve()),
        "fit_frames": sample_ids[fit_slots].tolist(),
        "validation_frames": sample_ids[validation_slots].tolist(),
        "reference_frame": reference_frame,
        "calibration": {
            "path": str(args.extrinsics_yaml.resolve()),
            **calibration_meta,
            "convention": "p_to = scale_factor * (rotation_matrix @ p_from) + translation_vector",
            "scale_factor": calibration_scale,
            "rotation_matrix": calibration_rotation.tolist(),
            "translation_vector": calibration_translation.tolist(),
        },
        "residual_alignment": {
            "type": "single_global_similarity",
            "estimated_on": "fit_frames",
            "scale": residual_scale,
            "rotation_matrix": residual_rotation.tolist(),
            "translation_vector": residual_translation.tolist(),
        },
        "anchor_errors": {
            "calibration_only_fit": anchor_stats(fit_slots, calibrated_anchor_errors),
            "calibration_only_validation": anchor_stats(validation_slots, calibrated_anchor_errors),
            "corrected_fit": anchor_stats(fit_slots, anchor_errors),
            "corrected_validation": anchor_stats(validation_slots, anchor_errors),
        },
        "markers": {label: data for (label, _), data in zip(MARKERS, diagnostics)},
        "face_keypoints": {
            label: {
                "number": len(MARKERS) + offset,
                "source": "mhr70",
                "keypoint_index": index,
            }
            for offset, (label, index) in enumerate(FACE_KEYPOINTS, start=1)
        },
    }
    map_path.write_text(json.dumps(payload, indent=2) + "\n")

    medians = np.array([item["median_validation_error_mm"] for item in diagnostics])
    print(f"Mapping: {map_path}")
    print(f"Trajectoires monde calibrees: {world_npy_path} {world_markers.shape}")
    print(f"Trajectoires monde corrigees: {npy_path} {output_markers.shape}")
    print(f"Frame de référence: {reference_frame}")
    print(f"Echelle residuelle globale: {residual_scale:.6f}")
    print(f"RMSE mediane des ancres corrigees: {np.median(anchor_rmse)*1000:.1f} mm")
    print(f"Erreur mediane de validation des marqueurs: moyenne={medians.mean():.1f} mm, max={medians.max():.1f} mm")
    for (label, _), item in zip(MARKERS, diagnostics):
        print(
            f"{item['number']:2d} {label:6s} vertex={item['vertex_index']:5d} "
            f"median={item['median_validation_error_mm']:5.1f} mm "
            f"p95={item['p95_validation_error_mm']:5.1f} mm"
        )


if __name__ == "__main__":
    main()
