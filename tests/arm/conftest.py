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


@pytest.fixture
def synthetic_trial(tmp_path):
    """Recorded views of 1012/Lifting rebuilt from archived landmarks (first 40 frames)."""
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    from rtcosmik.config_loader import settings

    names = list(settings.marker_names)
    table = np.genfromtxt(TRUTH_RUN / "markers.csv", delimiter=",", names=True)[:40]
    world = np.stack([np.stack([table[f"{n}_x"], table[f"{n}_y"], table[f"{n}_z"]], -1)
                      for n in names], 1)                                  # (F, M, 3)
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
    views = tmp_path / "views" / PARTICIPANT / f"{TASK}.npz"
    views.parent.mkdir(parents=True)
    np.savez_compressed(views, cameras=np.asarray(CAMERAS), keypoints=keypoints,
                        poses3d=poses, uncertainties=sigma)
    return {"dataset": COMFI, "views": views, "participant": PARTICIPANT, "task": TASK,
            "cameras": CAMERAS, "frames": len(world), "out_dir": tmp_path / "run"}
