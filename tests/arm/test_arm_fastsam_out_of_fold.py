import copy
import json

import numpy as np
import pytest

from .conftest import needs_comfi


def _settings(choice="out_of_fold"):
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.ik_type = "sbs"
    s.paper["landmark_map"] = choice
    s.paper["clock_offset"] = 0
    return s


def _run(trial, repo, settings, cameras=None):
    import sweep
    return sweep.run_fastsam(trial["dataset"], trial["participant"], trial["task"],
                             cameras or trial["cameras"], trial["out_dir"], settings, repo=repo)


@needs_comfi
def test_map_fitted_on_the_participant_is_refused(mesh_trial, clean_repo, monkeypatch):
    from rtcosmik import landmark_maps
    other_fold = landmark_maps.load("fastsam", 1)     # fitted on 1012
    monkeypatch.setattr(landmark_maps, "select", lambda *a, **k: other_fold)

    with pytest.raises(landmark_maps.LeakageError):
        _run(mesh_trial, clean_repo, _settings())
    assert not mesh_trial["out_dir"].exists()


def _markers(run_dir, name):
    table = np.genfromtxt(run_dir / "markers.csv", delimiter=",", names=True)
    return np.stack([table[f"{name}_{axis}"] for axis in "xyz"], -1)


@needs_comfi
def test_own_fold_map_runs_and_is_recorded(mesh_trial, clean_repo):
    frames, _, _ = _run(mesh_trial, clean_repo, _settings())

    assert frames == mesh_trial["frames"]
    info = json.loads((mesh_trial["out_dir"] / "run_info.json").read_text())
    assert info["landmark_map"]["front_end"] == "fastsam"
    assert info["landmark_map"]["fold"] == 0            # 1012 is in fold 0
    assert "1012" not in info["landmark_map"]["fitted_on"]
    assert info["landmark_map"]["in_sample"] is False
    assert len(info["landmark_map"]["sha256"]) == 64
    assert info["arm"]["cameras"] == [2, 4, 6]


@needs_comfi
def test_landmarks_are_read_at_the_fold_maps_vertices(mesh_trial, clean_repo):
    _run(mesh_trial, clean_repo, _settings())

    # C7 is planted at fold 0's vertex only; the other folds' vertex holds zeros.
    read = _markers(mesh_trial["out_dir"], "C7")[-1]
    assert np.abs(read - mesh_trial["world"]["C7"][-1]).max() < 0.01


@needs_comfi
def test_camera_without_a_stored_mesh_is_refused(mesh_trial, clean_repo):
    with pytest.raises(FileNotFoundError, match="no stored mesh"):
        _run(mesh_trial, clean_repo, _settings(), cameras=[0, 2])
    assert not mesh_trial["out_dir"].exists()


@needs_comfi
def test_3361_meshes_are_paired_with_the_cameras_they_hold(tmp_path, clean_repo):
    from .conftest import build_mesh_trial
    trial = build_mesh_trial(tmp_path, fold=2, participant="3361", cameras=(6, 0, 2))

    _run(trial, clean_repo, _settings())

    assert np.abs(_markers(trial["out_dir"], "RKNE")[-1] - trial["world"]["RKNE"][-1]).max() < 0.01
    info = json.loads((trial["out_dir"] / "run_info.json").read_text())
    assert (info["landmark_map"]["fold"], info["arm"]["cameras"]) == (2, [6, 0, 2])
    with pytest.raises(FileNotFoundError, match="no stored mesh"):
        _run(trial, clean_repo, _settings(), cameras=[2, 4, 6])


@needs_comfi
def test_out_of_fold_run_is_scored_causal_and_aligned(mesh_trial, clean_repo):
    from rtcosmik.paper.scoring import score_run
    settings = _settings()
    _run(mesh_trial, clean_repo, settings)

    row = score_run(mesh_trial["out_dir"], mesh_trial["dataset"], mesh_trial["participant"],
                    mesh_trial["task"], settings)

    assert row["scored_dof"] == 29
    assert np.isfinite([row["causal_rmse_mean"], row["aligned_rmse_mean"]]).all()


def test_face_keypoints_match_the_fold_maps():
    from rtcosmik import landmark_maps
    from rtcosmik.paper.fastsam_meshes import FACE_KEYPOINTS
    for fold in range(6):
        stored = json.loads(landmark_maps.path("fastsam", fold).read_text())["face_keypoints"]
        assert {name: entry["keypoint_index"] for name, entry in stored.items()} == FACE_KEYPOINTS
