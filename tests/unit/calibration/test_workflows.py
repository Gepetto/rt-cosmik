"""The two calibration scripts, run on images of a real two-camera rig.

The images are 8 of the rig's 17 checkerboard shots and its 4 robot-base wand
shots. Expected values are what the full set gave: baseline 0.8269 m, reference
camera at (-0.518, 1.917, 1.025) m from the robot base.
"""
import importlib.util
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from rtcosmik.calibration import files  # noqa: E402
from rtcosmik.camera import cam_utils  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA = os.path.join(os.path.dirname(__file__), "data")


def run_script(name, *arguments):
    """Run a calibration script's main() with these command-line arguments."""
    path = os.path.join(REPO, "scripts", "python", "calibration", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv, sys.argv = sys.argv, [path, "--from-images", "--no-show", *arguments]
    try:
        module.main()
    finally:
        sys.argv = argv


@pytest.fixture(scope="module")
def checkerboard_root(tmp_path_factory):
    """A calibration directory after calibrate_cameras.py."""
    root = str(tmp_path_factory.mktemp("cam_params"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(files, "camera_hardware_info",
                      lambda index: {"bus_info": f"port-{index}"})
        run_script("calibrate_cameras", "--cameras", "0", "2", "--cam-params", root,
                   "--images", DATA, "--labels", "0=left")
    return root


@pytest.fixture
def root(checkerboard_root, tmp_path):
    """A copy of the checkerboard calibration, for a test to add to."""
    return shutil.copytree(checkerboard_root, str(tmp_path / "cam_params"))


def test_checkerboard_writes_the_layout(checkerboard_root):
    written = sorted(os.path.relpath(os.path.join(folder, name), checkerboard_root)
                     for folder, _, names in os.walk(checkerboard_root) for name in names)
    assert written == ["cameras.yaml",
                       "extrinsics/cam_to_cam/camera_0_to_camera_2.yaml",
                       "intrinsics/camera_0_intrinsics.yaml",
                       "intrinsics/camera_2_intrinsics.yaml"]
    manifest = cam_utils.load_camera_manifest(checkerboard_root)
    assert manifest[0] == {"id": 0, "label": "left", "bus_info": "port-0"}
    assert manifest[2] == {"id": 2, "bus_info": "port-2"}


def test_checkerboard_places_the_second_camera(checkerboard_root):
    mtxs, _, _, _, translations = cam_utils.load_camera_parameters(
        checkerboard_root, camera_ids=[0, 2])
    assert np.linalg.norm(translations[1]) == pytest.approx(0.8269, abs=0.005)
    for mtx in mtxs:  # 1280 x 720 images
        assert mtx[0, 2] == pytest.approx(640, abs=100) and mtx[1, 2] == pytest.approx(360, abs=100)
        assert mtx[0, 0] == pytest.approx(mtx[1, 1], rel=0.02)


def test_robot_frame_anchors_the_reference_camera_only(root, capsys):
    run_script("set_world_frame", "--cameras", "0", "2", "--robot",
               "--cam-params", root, "--images", DATA)

    assert os.listdir(os.path.join(root, "extrinsics", "cam_to_world")) == ["camera_0"]
    _, position = cam_utils.load_world_transformation(root, ref_camera=0, required=True)
    assert np.allclose(position, [-0.518, 1.917, 1.025], atol=0.04)
    assert os.path.isfile(os.path.join(root, "world_frame_check.png"))
    assert "Using camera_0's own measurement" in capsys.readouterr().out
    # The toolbox still places camera 2 through the stereo pair.
    _, _, _, _, translations = cam_utils.load_camera_parameters(root, camera_ids=[0, 2])
    assert np.linalg.norm(translations[1]) == pytest.approx(0.8269, abs=0.005)


def test_ground_frame_runs_on_three_positions(root, tmp_path):
    # No ground pointings were recorded: three robot ones only show the path runs.
    for camera in (0, 2):
        source = os.path.join(DATA, f"images_world_cam_{camera}", "color")
        target = tmp_path / f"images_world_cam_{camera}" / "color"
        target.mkdir(parents=True)
        for name in ("img_0.png", "img_1.png", "img_2.png"):
            shutil.copy(os.path.join(source, name), target)

    run_script("set_world_frame", "--cameras", "0", "2", "--cam-params", root,
               "--images", str(tmp_path))

    assert cam_utils.cam_to_world_path(root, 0) is not None


def test_ground_frame_refuses_robot_images(root, capsys):
    with pytest.raises(SystemExit, match="no usable wand measurement"):
        run_script("set_world_frame", "--cameras", "0", "2", "--cam-params", root,
                   "--images", DATA)
    assert "4 images (need 3)" in capsys.readouterr().out
    assert cam_utils.cam_to_world_path(root, 0) is None


def test_world_frame_needs_intrinsics(tmp_path):
    with pytest.raises(SystemExit, match="missing intrinsics"):
        run_script("set_world_frame", "--cameras", "0", "2", "--robot",
                   "--cam-params", str(tmp_path), "--images", DATA)


def test_one_camera_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="at least two cameras"):
        run_script("calibrate_cameras", "--cameras", "0", "--cam-params", str(tmp_path),
                   "--images", DATA)
    assert os.listdir(tmp_path) == []
