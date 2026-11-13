"""Synthetic trials for the arm-evaluation seam."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
COMFI = Path("/root/workspace/COMFI")
PARTICIPANT, TASK, CAMERAS = "1012", "Lifting", [0, 2, 4, 6]
TRUTH_RUN = COMFI / "output" / "campaign" / PARTICIPANT / TASK / "nlf_0-2-4-6"
REFERENCE_RUN = TRUTH_RUN.parent / "mocap_reference"

for path in (REPO / "src", REPO / "scripts" / "python" / "paper"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

needs_comfi = pytest.mark.skipif(not TRUTH_RUN.exists(), reason="COMFI campaign outputs not mounted")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path):
    """A git repository with one commit and pipeline-shaped paths."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "module.py").write_text("x = 1\n")
    (repo / "settings.py").write_text("N = 7\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return repo


def build_trial(root, frames=40, lag=0, source=TRUTH_RUN):
    """Recorded views of 1012/Lifting rebuilt from a run's landmarks, delayed by ``lag`` frames.

    Landmarks ``source`` lacks (the reference has no face) come from the archived NLF run.
    """
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.config_loader import settings

    names = list(settings.marker_names)
    tables = [np.genfromtxt(run / "markers.csv", delimiter=",", names=True)[:frames]
              for run in (source, TRUTH_RUN)]
    table = lambda n: next(t for t in tables if f"{n}_x" in t.dtype.names)
    world = np.stack([np.stack([table(n)[f"{n}_{a}"] for a in "xyz"], -1) for n in names], 1)
    world = np.concatenate([np.repeat(world[:1], lag, 0), world[:len(world) - lag]])
    cam_dir = COMFI / "cam_params" / PARTICIPANT
    mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, CAMERAS)
    R, T = (np.asarray(a) for a in load_world_transformation(cam_dir, CAMERAS[0]))
    reference = (world - T) @ R                                            # world -> camera 0
    poses, keypoints = [], []
    for K, P in zip(mtxs, projections):
        P = np.asarray(P)
        cam = reference @ P[:, :3].T + P[:, 3]
        poses.append(cam)
        pix = cam @ np.asarray(K).T
        keypoints.append(pix[..., :2] / pix[..., 2:])
    poses = np.stack(poses, 1).astype(np.float32)                          # (F, C, M, 3)
    keypoints = np.stack(keypoints, 1).astype(np.float32)
    sigma = np.full(poses.shape[:3], 0.05, np.float32)
    views = root / "views" / PARTICIPANT / f"{TASK}.npz"
    views.parent.mkdir(parents=True)
    np.savez_compressed(views, cameras=np.asarray(CAMERAS), keypoints=keypoints,
                        poses3d=poses, uncertainties=sigma)
    return {"dataset": COMFI, "views": views, "participant": PARTICIPANT, "task": TASK,
            "cameras": CAMERAS, "frames": len(world), "out_dir": root / "run"}


@pytest.fixture
def synthetic_trial(tmp_path):
    return build_trial(tmp_path)


def build_mesh_trial(root, frames=40, fold=0, participant=PARTICIPANT, cameras=(2, 4, 6),
                     labels=(2, 4, 6)):
    """A FastSAM-3D mesh store holding one Lifting trial: 1012's archived landmarks planted
    at fold ``fold``'s vertices, seen by ``cameras`` and stored under ``labels``."""
    import json
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.paper import fastsam_meshes

    vertices_of = landmark_maps.load("fastsam", fold).vertices
    table = np.genfromtxt(TRUTH_RUN / "markers.csv", delimiter=",", names=True)[:frames]
    world = {n[:-2]: np.stack([table[f"{n[:-2]}_{a}"] for a in "xyz"], -1)
             for n in table.dtype.names if n.endswith("_x")}
    cam_dir = COMFI / "cam_params" / participant
    _, _, projections, _, _ = load_camera_parameters(cam_dir, list(cameras))
    R, T = (np.asarray(a) for a in load_world_transformation(cam_dir, cameras[0]))
    dataset = root / "dataset"
    store = dataset / fastsam_meshes.STORE
    store.mkdir(parents=True)
    for name in ("metadata", "cam_params", "output"):
        (dataset / name).symlink_to(COMFI / name)
    (store / fastsam_meshes.STORE_INDEX).write_text(json.dumps(
        {"subject_ids": {"someone": "1012"}, "task_directories": {"lifting": TASK}}))
    folder = "someone" if participant == "1012" else "unlisted"
    for label, P in zip(labels, projections):
        P = np.asarray(P)
        in_camera = lambda name: ((world[name] - T) @ R) @ P[:, :3].T + P[:, 3]
        vertices = np.zeros((frames, 18439, 3), np.float32)
        for name, vertex in vertices_of.items():
            if name in world:
                vertices[:, vertex] = in_camera(name)
        keypoints = np.zeros((frames, 70, 3), np.float32)
        for name, index in fastsam_meshes.FACE_KEYPOINTS.items():
            keypoints[:, index] = in_camera(name)
        result = store / folder / "lifting" / f"camera_{label}" / "calibrated"
        result.mkdir(parents=True)
        np.save(result / "vertices_cam.npy", vertices)
        np.save(result / "keypoints70_cam.npy", keypoints)
    return {"dataset": dataset, "participant": participant, "task": TASK,
            "cameras": list(cameras), "frames": frames, "out_dir": root / "run", "world": world}


@pytest.fixture
def mesh_trial(tmp_path):
    return build_mesh_trial(tmp_path)
