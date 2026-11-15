"""What the calibration writes is what the toolbox's loaders read."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from rtcosmik.calibration import files  # noqa: E402
from rtcosmik.camera import cam_utils  # noqa: E402

K0 = np.array([[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
K2 = np.array([[880.0, 0.0, 630.0], [0.0, 885.0, 350.0], [0.0, 0.0, 1.0]])
D0 = np.array([[0.1, -0.2, 0.001, 0.002, 0.05]])
D2 = np.zeros((1, 5))
# A quarter turn about z: p_2 = R p_0 + T.
R_02 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_02 = np.array([[0.5], [0.0], [0.1]])


def test_intrinsics_read_back(tmp_path):
    path = files.save_intrinsics(K0, D0, 0.21, 0, str(tmp_path))
    assert path == str(tmp_path / "intrinsics" / "camera_0_intrinsics.yaml")
    mtx, dist = cam_utils.load_cam_params(path)
    assert np.allclose(mtx, K0) and np.allclose(dist, D0)


def test_stereo_pose_reads_back_uninverted(tmp_path):
    path = files.save_stereo_pose(K0, D0, K2, D2, R_02, T_02, 0.6, 0, 2, str(tmp_path))
    assert path == str(tmp_path / "extrinsics" / "cam_to_cam" / "camera_0_to_camera_2.yaml")
    R, T = cam_utils.load_cam_to_cam(path)
    assert np.allclose(R, R_02) and np.allclose(T, [0.5, 0.0, 0.1])


def test_world_pose_reads_back_as_the_camera_in_the_world(tmp_path):
    path = files.save_world_pose(R_02, [1.0, 2.0, 1.5], 0, str(tmp_path))
    assert path == str(tmp_path / "extrinsics" / "cam_to_world" / "camera_0"
                       / "camera_0_extrinsics.yaml")
    R, T = cam_utils.load_world_transformation(str(tmp_path), ref_camera=0, required=True)
    assert np.allclose(R, R_02) and np.allclose(T, [1.0, 2.0, 1.5])
    placed = cam_utils.describe_camera_placement(str(tmp_path), camera_ids=[0, 2])
    assert list(placed) == [0] and np.allclose(placed[0], [1.0, 2.0, 1.5])


def test_written_rig_loads_as_stereo_pairs_plus_one_world_pose(tmp_path):
    root = str(tmp_path)
    files.save_intrinsics(K0, D0, 0.2, 0, root)
    files.save_intrinsics(K2, D2, 0.2, 2, root)
    files.save_stereo_pose(K0, D0, K2, D2, R_02, T_02, 0.6, 0, 2, root)
    files.save_world_pose(np.eye(3), [0.0, 0.0, 1.0], 0, root)

    mtxs, dists, projections, _, _ = cam_utils.load_camera_parameters(root, camera_ids=[0, 2])

    assert np.allclose(mtxs[1], K2) and np.allclose(dists[0], D0)
    assert np.allclose(projections[0], np.hstack([np.eye(3), np.zeros((3, 1))]))
    assert np.allclose(projections[1], np.hstack([R_02, T_02]))


def test_manifest_records_the_port_and_label_of_each_camera(tmp_path, monkeypatch):
    ports = {0: "usb-0000:00:14.0-7.1", 2: "usb-0000:00:14.0-8.1"}
    monkeypatch.setattr(files, "camera_bus_info", lambda index: ports.get(index))
    monkeypatch.setattr(cam_utils, "camera_bus_info", lambda index: ports.get(index))

    path = files.save_camera_manifest(str(tmp_path), [0, 2], labels={2: "front_right"})

    assert path == str(tmp_path / "cameras.yaml")
    manifest = cam_utils.load_camera_manifest(str(tmp_path))
    assert manifest[0]["bus_info"] == "usb-0000:00:14.0-7.1" and "label" not in manifest[0]
    assert manifest[2]["bus_info"] == "usb-0000:00:14.0-8.1"
    assert manifest[2]["label"] == "front_right"
    # Recabled: the device now at index 0 is on camera 2's port.
    ports.update({0: "usb-0000:00:14.0-8.1", 2: "usb-0000:00:14.0-7.1"})
    assert cam_utils.resolve_camera_ids(str(tmp_path), [0, 2]) == {0: 2, 2: 0}


def test_hardware_info_leaves_out_what_cannot_be_read(monkeypatch):
    monkeypatch.setattr(files, "camera_bus_info", lambda index: None)
    assert files.camera_hardware_info(987654) == {}


def test_relative_poses_are_read_back_and_replace_stale_world_poses(tmp_path):
    from rtcosmik.calibration.files import (save_intrinsics, save_relative_poses,
                                            save_world_pose)
    from rtcosmik.camera.cam_utils import load_camera_parameters
    cameras, root = [0, 2, 4], str(tmp_path)
    K, D = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]]), np.zeros((1, 5))
    turn = cv2.Rodrigues(np.array([0.1, -0.4, 0.2]))[0]
    poses = [(np.eye(3), np.zeros(3)), (turn, np.array([0.8, 0.0, 0.1])),
             (turn @ turn, np.array([-0.3, 0.2, 5.0]))]
    for camera in cameras:
        save_intrinsics(K, D, 0.0, camera, root)
        save_world_pose(np.eye(3), np.array([9.0, 9.0, 9.0]), camera, root)
    save_relative_poses(poses, [(K, D)] * 3, cameras, root)
    rotations, translations = load_camera_parameters(root, cameras)[3:]
    for (R, T), rotation, translation in zip(poses, rotations, translations):
        assert np.allclose(rotation, R)
        assert np.allclose(np.ravel(translation), T)
