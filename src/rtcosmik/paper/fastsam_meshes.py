"""FastSAM-3D markers read from the stored full meshes at a landmark map's vertices.

The store keeps every frame's whole mesh per camera, so a fold's map is applied by
reading other vertices: no inference is run again.
"""
import json
from pathlib import Path

import numpy as np

#: The store, under the dataset; its index names each participant's folder and tasks.
STORE = Path("fastsam") / "results_multicam"
STORE_INDEX = "cosmik_mhr_marker_map_17subjects_tv8_tv12.json"
#: Camera labels with stored meshes: camera 0's were not kept.
MESH_CAMERAS = (2, 4, 6)
#: The one participant whose folder the store's index does not list.
UNLISTED = "3361"
#: Face landmarks are MHR70 keypoints, fixed a priori, not fitted.
FACE_KEYPOINTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4}


def result_dir(dataset, participant, task, label):
    """Folder holding one trial's meshes for the camera labelled ``label``."""
    store = Path(dataset) / STORE
    index = json.loads((store / STORE_INDEX).read_text())
    ids = index["subject_ids"]
    owner = lambda name: next((ids[c] for c in (name, name.rstrip("_"), name + "_") if c in ids),
                              UNLISTED)
    folders = [p for p in sorted(store.iterdir()) if p.is_dir() and owner(p.name) == participant]
    tasks = [raw for raw, name in index["task_directories"].items() if name == task]
    if len(folders) != 1 or len(tasks) != 1:
        raise FileNotFoundError(f"no single mesh folder for {participant}/{task} in {store}")
    return folders[0] / tasks[0] / f"camera_{label}" / "calibrated"


def load_mesh_markers(directory, landmark_map):
    """One camera's markers at the map's vertices: ``(names, xyz, valid)``, like the CSV reader."""
    directory = Path(directory)
    vertices = np.load(directory / "vertices_cam.npy", mmap_mode="r")
    keypoints = np.load(directory / "keypoints70_cam.npy", mmap_mode="r")
    if len(keypoints) != len(vertices) or keypoints.shape[1:] != (70, 3):
        raise ValueError(f"unexpected keypoints in {directory}: {keypoints.shape}")
    names = list(landmark_map.vertices) + list(FACE_KEYPOINTS)
    xyz = np.concatenate([vertices[:, list(landmark_map.vertices.values())],
                          keypoints[:, list(FACE_KEYPOINTS.values())]], axis=1).astype(float)
    frames = directory / "frame_ids.npy"
    if frames.exists() and not np.array_equal(np.load(frames).ravel(), np.arange(len(xyz))):
        raise ValueError(f"{directory}: rows are not consecutive frames")
    flags = directory / "valid.npy"
    valid = np.load(flags).astype(bool).ravel() if flags.exists() else np.ones(len(xyz), bool)
    valid = valid & np.isfinite(xyz).all(axis=(1, 2))
    xyz[~valid] = np.nan
    return names, xyz, valid.astype(int)


def load_views(dataset, participant, task, labels, landmark_map):
    """Markers of the cameras labelled ``labels``, refusing a label with no stored mesh."""
    missing = [label for label in labels if label not in MESH_CAMERAS]
    if missing:
        raise FileNotFoundError(f"no stored mesh for {participant} under camera label(s) "
                                f"{missing}: the store holds {list(MESH_CAMERAS)}")
    return [load_mesh_markers(result_dir(dataset, participant, task, label), landmark_map)
            for label in labels]
