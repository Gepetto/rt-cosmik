"""Writing a calibration in the layout :mod:`rtcosmik.camera.cam_utils` reads."""
import os

import cv2 as cv
import numpy as np
import yaml

from rtcosmik.camera.cam_utils import (
    camera_bus_info, camera_manifest_path, intrinsics_path)


def stereo_pose_path(root, cam_a, cam_b):
    """Path of the stereo result relating two cameras."""
    return os.path.join(root, "extrinsics", "cam_to_cam",
                        f"camera_{cam_a}_to_camera_{cam_b}.yaml")


def world_pose_path(root, camera_id):
    """Path of one camera's pose in the world."""
    return os.path.join(root, "extrinsics", "cam_to_world",
                        f"camera_{camera_id}", f"camera_{camera_id}_extrinsics.yaml")


def _write_opencv(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    storage = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    for key, value in values.items():
        storage.write(key, value)
    storage.release()
    return path


def save_intrinsics(mtx, dist, reproj, camera_id, root):
    """Write one camera's matrix, distortion and reprojection error (px)."""
    return _write_opencv(intrinsics_path(root, camera_id),
                         {"K": mtx, "D": dist, "reproj": reproj})


def save_stereo_pose(mtx1, dist1, mtx2, dist2, R, T, rmse, cam_a, cam_b, root):
    """Write a stereo pair as ``cv2.stereoCalibrate`` gives it: ``p_b = R @ p_a + T``."""
    return _write_opencv(stereo_pose_path(root, cam_a, cam_b),
                         {"K1": mtx1, "D1": dist1, "K2": mtx2, "D2": dist2,
                          "R": R, "T": T, "rmse": rmse})


def save_world_pose(world_R_cam, world_T_cam, camera_id, root):
    """Write a camera's pose in the world: ``p_world = R @ p_cam + T``, not inverted here."""
    R = np.asarray(world_R_cam, dtype=float).reshape(3, 3)
    T = np.asarray(world_T_cam, dtype=float).reshape(3)
    path = world_pose_path(root, camera_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"camera_extrinsics": {
        "frame_from": f"camera_{camera_id}",
        "frame_to": "world",
        "rotation_matrix": [[float(v) for v in row] for row in R],
        "translation_vector": [float(v) for v in T],
    }}
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    return path


def _read_text(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return handle.read().strip()


def camera_hardware_info(index):
    """What can be read about the device behind a v4l2 index; unknown fields are left out."""
    sysfs = f"/sys/class/video4linux/video{index}"
    usb = os.path.realpath(os.path.join(sysfs, "device", ".."))
    info = {
        "model": _read_text(os.path.join(sysfs, "name")),
        "serial": _read_text(os.path.join(usb, "serial")),
        "product": _read_text(os.path.join(usb, "product")),
        "manufacturer": _read_text(os.path.join(usb, "manufacturer")),
    }
    vendor = _read_text(os.path.join(usb, "idVendor"))
    product = _read_text(os.path.join(usb, "idProduct"))
    if vendor and product:
        info["vendor_product"] = f"{vendor}:{product}"
    # The USB port is what tells identical cameras apart.
    info["bus_info"] = camera_bus_info(index)
    return {key: value for key, value in info.items() if value}


def save_camera_manifest(root, camera_ids, labels=None):
    """Write ``cameras.yaml``: the hardware, and optional label, behind each camera id."""
    labels = labels or {}
    entries = []
    for camera_id in camera_ids:
        entry = {"id": int(camera_id)}
        if camera_id in labels:
            entry["label"] = labels[camera_id]
        entry.update(camera_hardware_info(camera_id))
        entries.append(entry)
    path = camera_manifest_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump({"cameras": entries}, handle,
                       default_flow_style=False, sort_keys=False)
    return path
