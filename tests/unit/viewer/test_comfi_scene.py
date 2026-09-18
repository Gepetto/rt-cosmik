"""The COMFI scene: what it resolves from the dataset, and that it degrades.

A stub stands in for meshcat so nothing needs a browser or a server: it records
which nodes received an object, a transform or a property. The dataset tests
run on one COMFI trial when the dataset is mounted (``COMFI_ROOT``, default
/root/workspace/COMFI) and are skipped otherwise.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.viewer.comfi_scene import ComfiScene, TrialAssets      # noqa: E402

COMFI = Path(os.environ.get("COMFI_ROOT", "/root/workspace/COMFI"))
needs_comfi = pytest.mark.skipif(not (COMFI / "cam_params" / "1012").is_dir(),
                                 reason="COMFI dataset not mounted")


class StubViewer:
    """Records meshcat calls by node path."""

    def __init__(self, log=None, path=""):
        self.log = {} if log is None else log
        self.path = path
        self.window = self

    def __getitem__(self, name):
        name = str(name)
        path = name if name.startswith("/") else f"{self.path}/{name}"
        return StubViewer(self.log, path)

    def _record(self, kind, value=None):
        self.log.setdefault(self.path, []).append((kind, value))

    def set_object(self, obj, material=None):
        self._record("object", obj)

    def set_transform(self, matrix=np.eye(4)):
        self._record("transform", np.asarray(matrix))

    def set_property(self, key, value):
        self._record(key, value)

    def delete(self):
        self._record("delete")

    def send(self, command):            # pinocchio sends DAE meshes directly
        self._record("send", command)

    def nodes(self, prefix):
        return sorted(p for p in self.log if p.startswith(prefix))


def test_without_a_dataset_the_scene_is_a_bare_floor():
    assets = TrialAssets.resolve(None, None, None)
    assert assets.missing == ["dataset"]
    viewer = StubViewer()
    scene = ComfiScene(viewer, assets)
    assert scene.robot is None
    assert not viewer.nodes("/scene")                       # no table, robot or cameras
    assert viewer.log["/Grid"][0][0] == "transform"         # the floor is still there


def test_missing_assets_are_listed_not_raised(tmp_path):
    assets = TrialAssets.resolve(tmp_path, "0000", "RobotWelding")
    assert any(m.startswith("camera 0") for m in assets.missing)
    assert "robot base pose" in assets.missing
    assert assets.cameras == {} and assets.robot_base is None
    ComfiScene(StubViewer(), assets)                        # and it still draws


@needs_comfi
def test_robot_trial_resolves_every_asset():
    assets = TrialAssets.resolve(COMFI, "1012", "RobotWelding")
    assert sorted(assets.cameras) == [0, 2, 4, 6]
    for T in assets.cameras.values():
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
    assert assets.robot_base is not None and assets.table is not None
    assert len(assets.force_plates) == 5
    frames = sorted(assets.robot_joints)
    assert len(frames) > 1000 and all(len(q) == 7 for q in assets.robot_joints.values())
    assert assets.missing == []


@needs_comfi
def test_tasks_without_a_robot_get_neither_robot_nor_table():
    assets = TrialAssets.resolve(COMFI, "1012", "Lifting")
    assert assets.robot_base is None and assets.table is None and assets.robot_joints == {}
    assert len(assets.cameras) == 4 and len(assets.force_plates) == 5


@needs_comfi
def test_the_robot_table_moves_with_the_robot():
    """The robot sits on its table, and was moved between sessions: the table
    follows the base, level, with its top at the base's height."""
    for participant in ("1012", "1118"):
        assets = TrialAssets.resolve(COMFI, participant, "RobotWelding")
        base, table = assets.robot_base, assets.table
        assert table["height"] == pytest.approx(base[2, 3])
        np.testing.assert_allclose(table["pose"][:3, 2], [0, 0, 1])              # level
        local = np.linalg.inv(table["pose"]) @ np.r_[base[:3, 3], 1.0]
        length, width = table["size"]
        assert abs(local[0]) < length / 2 and abs(local[1]) < width / 2      # base on the top


@needs_comfi
def test_work_bench_tasks_get_the_bench():
    for task in ("Screwing", "PolishingSat"):
        table = TrialAssets.resolve(COMFI, "1012", task).table
        assert table is not None and table["height"] < 0.9


@needs_comfi
def test_scene_draws_the_room_and_follows_the_robot():
    import example_robot_data as robex
    assets = TrialAssets.resolve(COMFI, "1012", "RobotWelding")
    viewer = StubViewer()
    scene = ComfiScene(viewer, assets)
    assert viewer.nodes("/scene/table/top") and len(viewer.nodes("/scene/table/leg_")) == 4
    assert len(viewer.nodes("/scene/force_plates/fp")) == 5
    assert len([n for n in viewer.nodes("/scene/cameras/camera_") if n.endswith("body")]) == 4
    assert viewer.nodes("/scene/robot")

    human = robex.human.HumanLoader(height=1.70, weight=51.0, gender="f").robot
    scene.add_body("estimate", human.model, human.visual_model)
    assert viewer.nodes("/bodies/estimate")

    frame = sorted(assets.robot_joints)[len(assets.robot_joints) // 2]
    link = [n for n in viewer.nodes("/scene/robot/visuals") if "link4" in n][0]
    before = len(viewer.log[link])
    scene.show(frame, {"estimate": human.q0})
    moved = viewer.log[link][before:]
    assert moved and moved[-1][0] == "transform"
