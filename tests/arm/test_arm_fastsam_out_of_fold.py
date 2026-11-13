import copy
import json

import numpy as np
import pytest

from .conftest import COMFI, needs_comfi, needs_mesh_store


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
def test_fold_map_fitted_on_the_participant_is_refused(mesh_trial, clean_repo, monkeypatch):
    from rtcosmik import landmark_maps
    # A wrong fold assignment: the real select then hands 1012 fold 1's map, fitted on 1012.
    monkeypatch.setattr(landmark_maps, "fold_of", lambda participant: 1)

    with pytest.raises(landmark_maps.LeakageError, match="map of fold 1 "):
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
def test_3361_runs_a_configuration_with_the_cameras_its_meshes_hold(tmp_path, clean_repo):
    from .conftest import build_mesh_trial
    trial = build_mesh_trial(tmp_path, fold=2, participant="3361", cameras=(6, 0, 2))

    _run(trial, clean_repo, _settings(), cameras=[2, 4, 6])

    assert np.abs(_markers(trial["out_dir"], "RKNE")[-1] - trial["world"]["RKNE"][-1]).max() < 0.01
    info = json.loads((trial["out_dir"] / "run_info.json").read_text())
    assert info["landmark_map"]["fold"] == 2
    assert (info["arm"]["configuration"], info["arm"]["cameras"]) == ([2, 4, 6], [6, 0, 2])


def test_mesh_cameras_mirror_3361_only():
    from rtcosmik.paper.fastsam_source import mesh_cameras
    assert mesh_cameras("3361", [2, 4, 6]) == [6, 0, 2]
    assert [mesh_cameras("3361", c) for c in ([2], [4, 6], [2, 4])] == [[6], [0, 2], [6, 0]]
    assert mesh_cameras("4687", [2, 4, 6]) == [2, 4, 6]


@needs_comfi
def test_sweep_summary_keeps_the_configuration_and_records_the_cameras_used(
        tmp_path, clean_repo, monkeypatch):
    import csv
    import functools
    import sys
    import sweep
    from rtcosmik import config_loader, provenance
    from .conftest import build_mesh_trial
    trial = build_mesh_trial(tmp_path, fold=2, participant="3361", cameras=(6, 0, 2))
    for listing in ("mmpose/output", "mocap/aligned"):
        (trial["dataset"] / listing / "3361" / trial["task"]).mkdir(parents=True)
    summary = tmp_path / "fastsam_2-4-6.csv"
    monkeypatch.setattr(config_loader, "settings", _settings())
    monkeypatch.setattr(provenance, "require_clean", lambda *a, **k: None)
    monkeypatch.setitem(sweep.ARMS, "fastsam",
                        functools.partial(sweep.run_fastsam, repo=clean_repo))
    monkeypatch.setattr(sys, "argv", [
        "sweep.py", "--arm", "fastsam", "--dataset", str(trial["dataset"]), "--cameras", "2", "4",
        "6", "--summary", str(summary), "--tag", "fastsam_2-4-6", "--output-dir",
        str(tmp_path / "runs")])

    sweep.main()
    sweep.main()                                         # resumed: the row is not run again

    rows = list(csv.DictReader(open(summary)))
    assert [(r["participant"], r["cameras"], r["cameras_used"], r["status"]) for r in rows] == [
        ("3361", "2-4-6", "6-0-2", "ok")]
    assert (tmp_path / "runs" / "3361" / trial["task"] / "fastsam_2-4-6" / "run_info.json").exists()


@needs_comfi
def test_submitted_map_reads_the_exports_and_is_recorded_in_sample(tmp_path, clean_repo):
    from .conftest import build_export_trial
    trial = build_export_trial(tmp_path)

    frames, _, _ = _run(trial, clean_repo, _settings("submitted"))

    assert frames == trial["frames"]
    assert np.abs(_markers(trial["out_dir"], "RKNE")[-1] - trial["world"]["RKNE"][-1]).max() < 0.01
    info = json.loads((trial["out_dir"] / "run_info.json").read_text())
    assert (info["landmark_map"]["fold"], info["landmark_map"]["in_sample"]) == (None, True)
    assert info["arm"]["cameras"] == info["arm"]["configuration"] == [0, 2]


@needs_comfi
def test_invalid_mesh_frames_are_dropped(tmp_path, clean_repo):
    from .conftest import build_mesh_trial
    trial = build_mesh_trial(tmp_path, invalid=(5,))

    frames, _, _ = _run(trial, clean_repo, _settings(), cameras=[2])

    table = np.genfromtxt(trial["out_dir"] / "markers.csv", delimiter=",", names=True)
    assert frames == trial["frames"] - 1
    assert 5 not in table["Frame_0"] and {4, 6} <= set(table["Frame_0"])


@needs_comfi
def test_non_consecutive_mesh_frames_are_refused(tmp_path, clean_repo):
    from .conftest import build_mesh_trial
    ids = np.arange(40)
    ids[20:] += 1                                        # a frame is missing from the store
    trial = build_mesh_trial(tmp_path, frame_ids=ids)

    with pytest.raises(ValueError, match="not consecutive frames"):
        _run(trial, clean_repo, _settings())
    assert not trial["out_dir"].exists()


@pytest.mark.parametrize("missing", ["frame_ids.npy", "valid.npy"])
@needs_comfi
def test_mesh_folder_without_frame_ids_or_validity_is_refused(mesh_trial, clean_repo, missing):
    from rtcosmik.paper import fastsam_meshes
    fastsam_meshes.result_dir(mesh_trial["dataset"], "1012", "Lifting", 2).joinpath(missing).unlink()

    with pytest.raises(FileNotFoundError):
        _run(mesh_trial, clean_repo, _settings())


@needs_mesh_store
def test_3361_stored_meshes_match_the_reference_only_with_their_paired_cameras():
    from rtcosmik import landmark_maps
    from rtcosmik.camera.cam_utils import load_world_transformation
    from rtcosmik.paper import fastsam_meshes
    from rtcosmik.paper.fastsam_source import mesh_cameras
    frames = np.arange(100, 400, 20)
    # Screwing: on Lifting the two cameras of a stereo pair fit label 4 equally well.
    reference = np.genfromtxt(COMFI / "output" / "campaign" / "3361" / "Screwing"
                              / "mocap_reference" / "markers.csv", delimiter=",", names=True)
    landmark_map = landmark_maps.load("fastsam", 2)
    names = [n for n in landmark_map.vertices if f"{n}_x" in reference.dtype.names]
    truth = np.stack([np.stack([reference[f"{n}_{a}"][frames] for a in "xyz"], -1)
                      for n in names], 1)

    for label in fastsam_meshes.MESH_CAMERAS:
        directory = fastsam_meshes.result_dir(COMFI, "3361", "Screwing", label)
        vertices = np.load(directory / "vertices_cam.npy", mmap_mode="r")
        seen = np.stack([vertices[f][[landmark_map.vertices[n] for n in names]]
                         for f in frames]).astype(float)
        error = {}
        for camera in (0, 2, 4, 6):
            R, T = (np.asarray(a) for a in load_world_transformation(
                COMFI / "cam_params" / "3361", camera))
            error[camera] = np.median(np.linalg.norm(seen @ R.T + T - truth, axis=-1))
        paired, = mesh_cameras("3361", [label])
        others = [e for camera, e in error.items() if camera != paired]
        assert error[paired] < 0.15 and min(others) > 2 * error[paired], (label, error)
        assert error[label] > 10 * error[paired], (label, error)    # the labels taken at face value


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
