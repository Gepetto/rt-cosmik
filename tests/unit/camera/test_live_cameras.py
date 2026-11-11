"""Which attached device each requested camera is, without any hardware.

v4l2 enumeration and USB port lookups are replaced by fixed answers, so these
run anywhere: python3 -m pytest tests/unit/camera/test_live_cameras.py
"""
import pytest
import yaml

from rtcosmik.camera import cam_utils
from rtcosmik.camera.cam_utils import anchor_first, select_live_cameras


def attach(monkeypatch, devices):
    """Pretend ``{index: (name, usb_port)}`` are what v4l2 reports."""
    monkeypatch.setattr(cam_utils, "list_cameras",
                        lambda: {index: name for index, (name, _) in devices.items()})
    monkeypatch.setattr(cam_utils, "camera_bus_info",
                        lambda index: devices[index][1] if index in devices else None)


def write_manifest(root, ports):
    """A cameras.yaml recording the USB port each camera id was calibrated on."""
    entries = [{"id": camera_id, "bus_info": port} for camera_id, port in ports.items()]
    (root / "cameras.yaml").write_text(yaml.safe_dump({"cameras": entries}))


def write_world_pose(root, camera_id):
    directory = root / "extrinsics" / "cam_to_world" / f"camera_{camera_id}"
    directory.mkdir(parents=True)
    (directory / f"camera_{camera_id}_extrinsics.yaml").write_text(yaml.safe_dump(
        {"camera_extrinsics": {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               "translation_vector": [0, 0, 1]}}))


def test_without_manifest_ids_are_device_numbers_and_extra_devices_stay_closed(
        monkeypatch, tmp_path):
    attach(monkeypatch, {0: ("Integrated Webcam", "usb-0-1"),
                         2: ("Cam", "usb-0-2"), 4: ("Cam", "usb-0-3")})
    assert select_live_cameras(tmp_path, [2, 4]) == [2, 4]


def test_requested_order_is_kept(monkeypatch, tmp_path):
    attach(monkeypatch, {2: ("Cam", "usb-0-2"), 4: ("Cam", "usb-0-3")})
    assert select_live_cameras(tmp_path, [4, 2]) == [4, 2]


def test_manifest_follows_cameras_across_a_recabling(monkeypatch, tmp_path):
    write_manifest(tmp_path, {6: "usb-0-7.1", 8: "usb-0-8.1"})
    # camera_8 now enumerates first, and a webcam took another index.
    attach(monkeypatch, {0: ("Integrated Webcam", "usb-0-5"),
                         2: ("Cam", "usb-0-8.1"), 4: ("Cam", "usb-0-7.1")})
    assert select_live_cameras(tmp_path, [6, 8]) == [4, 2]


def test_device_holding_a_calibrated_number_on_another_port_is_not_that_camera(
        monkeypatch, tmp_path):
    write_manifest(tmp_path, {0: "usb-0-7.1", 2: "usb-0-8.1"})
    # The built-in webcam got /dev/video0; the calibrated camera_0 moved to 3.
    attach(monkeypatch, {0: ("Integrated Webcam", "usb-0-5"),
                         3: ("Cam", "usb-0-7.1"), 5: ("Cam", "usb-0-8.1")})
    assert select_live_cameras(tmp_path, [0, 2]) == [3, 5]


def test_ids_absent_from_the_manifest_fall_back_to_device_numbers(monkeypatch, tmp_path):
    write_manifest(tmp_path, {6: "usb-0-7.1"})
    attach(monkeypatch, {0: ("Cam", "usb-0-2"), 4: ("Cam", "usb-0-7.1")})
    assert select_live_cameras(tmp_path, [0, 6]) == [0, 4]


def test_missing_camera_is_named_with_what_is_attached(monkeypatch, tmp_path):
    attach(monkeypatch, {0: ("Integrated Webcam", "usb-0-5"), 2: ("Cam", "usb-0-2")})
    with pytest.raises(RuntimeError) as error:
        select_live_cameras(tmp_path, [2, 4])
    message = str(error.value)
    assert "[4]" in message
    assert "/dev/video0 (Integrated Webcam) is camera_0" in message
    assert "/dev/video2 (Cam) is camera_2" in message


def test_no_camera_attached(monkeypatch, tmp_path):
    attach(monkeypatch, {})
    with pytest.raises(RuntimeError, match="v4l2-ctl"):
        select_live_cameras(tmp_path, [0])


def test_anchored_camera_becomes_the_reference(tmp_path):
    write_world_pose(tmp_path, 4)
    assert anchor_first(tmp_path, [2, 4, 6]) == [4, 2, 6]


def test_order_kept_when_the_first_camera_is_anchored(tmp_path):
    write_world_pose(tmp_path, 2)
    write_world_pose(tmp_path, 4)
    assert anchor_first(tmp_path, [2, 4]) == [2, 4]


def test_order_kept_without_any_anchor(tmp_path):
    assert anchor_first(tmp_path, [2, 4]) == [2, 4]
