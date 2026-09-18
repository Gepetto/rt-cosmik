"""The COMFI capture scene in meshcat: floor, cameras, table, robot and bodies.

Ported from the dataset's own example code (Gepetto/comfi-examples,
``scripts/visualization/viz_all_data.py`` and ``comfi_examples/viz_utils.py``:
``define_scene``, ``draw_table``, ``add_markers_to_meshcat``,
``make_visuals_gray``, ``load_robot_panda``, ``load_robot_base_pose``), so a
replayed trial is shown in the room it was recorded in. Everything is resolved
from the dataset root, the participant id and the task name:

==================  =========================================================
floor               meshcat's grid at z = 0 (the mocap world frame is z up)
cameras             ``cam_params/<id>/extrinsics/cam_to_world/camera_<k>``, the
                    same calibration the pipeline reads
force plates        the five plates of comfi-examples, flush with the floor
                    (only the plates: forces are not drawn)
tables              two different tables, neither recorded by the dataset: the
                    robot is mounted on its own table, placed here relative to
                    the robot base (it moves with the robot between sessions);
                    the tasks done at a work bench use comfi-examples' fixed
                    table pose, at the bench's height
robot               the Franka Panda (example-robot-data), placed by
                    ``robot/robot_in_world/<id>/robot_base_pose.yaml`` and
                    driven by ``robot/aligned/<id>/<id>_<task>.csv``, matched to
                    video frames by camera-0 timestamp
bodies              any number of human models, each with its own colour and
                    opacity; the model itself is the caller's (the pipeline's
                    calibrated model, or a run's model rebuilt for a figure)
markers             point sets, as spheres or a point cloud
==================  =========================================================

The force arrows of the original viewer are not ported.

Every dataset asset is optional. :meth:`TrialAssets.resolve` records what it
could not find and the scene simply leaves it out, so the same code draws a
bare floor for a live run on cameras that belong to no dataset.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: The robot's table (m): long side along the robot's x axis, the robot base
#: this far from its back edge and centred across it, top at the base's height.
#: Sized and placed by projecting it onto the videos of several participants;
#: the width is kept just inside the operators' closest approach (the front of
#: the pelvis and the knees come within 0.37 m of the robot's axis) so the body
#: does not sink into it.
ROBOT_TABLE_LENGTH, ROBOT_TABLE_WIDTH, ROBOT_TABLE_BACK = 1.20, 0.70, 0.18
#: The work bench of the tasks below: comfi-examples' table pose and size (x, y
#: extent), at the bench's height, checked the same way.
WORK_TABLE_TASKS = ("Screwing", "ScrewingSat", "Polishing", "PolishingSat",
                    "Hammering", "HammeringSat", "Welding", "WeldingSat")
WORK_TABLE_CENTRE_XY, WORK_TABLE_SIZE, WORK_TABLE_HEIGHT = (0.9, -0.6), (0.90, 1.80), 0.75
TABLE_THICKNESS, TABLE_LEG, TABLE_INSET = 0.04, 0.05, 0.05
#: Force plates, from comfi-examples: (x, y) size and centre (m), on the floor.
FORCE_PLATES = (((0.5, 0.6), (-0.83, -0.3)), ((0.5, 0.6), (-0.25, -0.3)), ((0.5, 0.6), (0.39, -0.3)),
                ((0.9, 1.8), (-1.68, -0.3)), ((0.5, 0.6), (-0.25, 0.3)))
FORCE_PLATE_THICKNESS = 0.01
FORCE_PLATE_RGBA = (0.5, 0.5, 0.5, 1.0)
#: Default colours, from comfi-examples: brown top, grey legs.
TABLE_TOP_RGBA = (0.80, 0.60, 0.40, 1.0)
TABLE_LEG_RGBA = (0.45, 0.45, 0.45, 1.0)
#: The two camera supports of COMFI hold cameras 0-2 and 4-6.
CAMERA_SUPPORTS = ((0, 2), (4, 6))


def task_has_robot(task):
    return task is not None and "robot" in task.lower()


def robot_joints_by_frame(dataset, participant, task):
    """Video frame index -> the Panda's 7 joint positions, for one trial.

    COMFI's robot rows carry camera 0's own timestamps, so a row is matched to
    the video frame whose timestamp is within 1 ms; recordings that start late
    simply cover fewer frames. Empty when the trial has no robot states.
    """
    import pandas as pd
    path = Path(dataset) / "robot" / "aligned" / participant / f"{participant}_{task}.csv"
    stamps = Path(dataset) / "videos" / participant / task / "camera_0_timestamps.csv"
    if not (path.is_file() and stamps.is_file()):
        return {}
    robot = pd.read_csv(path)
    if robot.empty:
        return {}
    video = pd.read_csv(stamps)
    video_ns = pd.to_datetime(video["timestamp"], utc=True).astype("int64").to_numpy()
    robot_ns = pd.to_datetime(robot["_cam_time"], utc=True).astype("int64").to_numpy()
    index = np.clip(np.searchsorted(video_ns, robot_ns), 0, len(video_ns) - 1)
    exact = np.abs(video_ns[index] - robot_ns) < 1_000_000
    joints = robot[[f"panda_joint{i}_position[rad]" for i in range(1, 8)]].to_numpy(float)
    return {int(video["frame_index"].iloc[i]): joints[k]
            for k, i in enumerate(index) if exact[k] and np.isfinite(joints[k]).all()}


def robot_table(world_T_robot):
    """The robot's table, from the robot base pose: only the base's heading and
    position are used, so a slightly tilted base calibration leaves it level."""
    R, t = world_T_robot[:3, :3], world_T_robot[:3, 3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:2, :2] = [[c, -s], [s, c]]
    pose[:2, 3] = t[:2] + (ROBOT_TABLE_LENGTH / 2 - ROBOT_TABLE_BACK) * np.array([c, s])
    return {"pose": pose, "size": (ROBOT_TABLE_LENGTH, ROBOT_TABLE_WIDTH), "height": float(t[2])}


@dataclass
class TrialAssets:
    """What the scene of one trial needs, resolved from the dataset layout."""

    participant: str = None
    task: str = None
    cameras: dict = field(default_factory=dict)      # id -> 4x4 world_T_camera (OpenCV axes)
    intrinsics: dict = field(default_factory=dict)   # id -> (K, dist)
    robot_base: np.ndarray = None                    # 4x4 world_T_robot
    robot_joints: dict = field(default_factory=dict)  # video frame -> 7 joint positions
    table: dict = None      # pose (4x4, floor under the top's centre, x along size[0]), size, height
    force_plates: tuple = ()                         # ((size x, size y), (centre x, centre y))
    missing: list = field(default_factory=list)

    @classmethod
    def resolve(cls, dataset, participant, task, camera_ids=(0, 2, 4, 6)):
        """Find every asset of a trial; whatever is absent is left out and listed."""
        assets = cls(participant=str(participant), task=task)
        if dataset is None or participant is None:
            assets.missing.append("dataset")
            return assets
        root = Path(dataset)

        from rtcosmik.camera.cam_utils import load_cam_params, intrinsics_path
        from rtcosmik.camera.cam_utils import load_world_transformation
        cam_root = root / "cam_params" / str(participant)
        for k in camera_ids:
            try:
                R, t = load_world_transformation(cam_root, k, required=True)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(t, float).ravel()
                assets.cameras[k] = T
                K, D = load_cam_params(intrinsics_path(cam_root, k))
                assets.intrinsics[k] = (np.asarray(K, float), np.asarray(D, float).ravel())
            except Exception as exc:
                assets.missing.append(f"camera {k} ({type(exc).__name__})")

        if task_has_robot(task):
            base = root / "robot" / "robot_in_world" / str(participant) / "robot_base_pose.yaml"
            if base.is_file():
                import yaml
                pose = yaml.safe_load(base.read_text())["world_T_robot"]
                assets.robot_base = np.asarray(pose["matrix_4x4"], dtype=float)
                assets.robot_joints = robot_joints_by_frame(root, str(participant), task)
                if not assets.robot_joints:
                    assets.missing.append("robot joint states")
            else:
                assets.missing.append("robot base pose")
        if assets.robot_base is not None:
            assets.table = robot_table(assets.robot_base)
        elif task in WORK_TABLE_TASKS:
            pose = np.eye(4)
            pose[:2, 3] = WORK_TABLE_CENTRE_XY
            assets.table = {"pose": pose, "size": WORK_TABLE_SIZE, "height": WORK_TABLE_HEIGHT}
        if (root / "forces").is_dir():
            assets.force_plates = FORCE_PLATES
        if assets.missing:
            LOGGER.info("[SCENE] %s/%s: not found, left out: %s", participant, task,
                        ", ".join(assets.missing))
        return assets


def _material(rgba):
    import meshcat.geometry as g
    colour = int(rgba[0] * 255) * 256 ** 2 + int(rgba[1] * 255) * 256 + int(rgba[2] * 255)
    return g.MeshLambertMaterial(color=colour, opacity=float(rgba[3]),
                                 transparent=float(rgba[3]) < 1.0)


def _recolour(geometry_model, rgba):
    """comfi-examples' make_visuals_gray, for any colour."""
    for go in geometry_model.geometryObjects:
        go.overrideMaterial = True
        go.meshColor = np.asarray(rgba, dtype=float)


class ComfiScene:
    """A meshcat scene of one COMFI trial.

    Typical use, one body tracked live::

        scene = ComfiScene(meshcat.Visualizer(), TrialAssets.resolve(root, "1012", "RobotWelding"))
        scene.add_body("estimate", model, visual_model)
        for frame, q in enumerate(stream):
            scene.show(frame, {"estimate": q})

    Args:
        viewer: a ``meshcat.Visualizer``.
        assets: a :class:`TrialAssets`, or None for a bare floor.
        show_cameras: draw the cameras and their supports.
        table_rgba, robot_rgba: override the table and robot colours (e.g. a
            light grey so bodies stand out); None keeps the dataset's look.
        background: RGB of the (flat) background.
    """

    def __init__(self, viewer, assets=None, show_cameras=True, table_rgba=None, robot_rgba=None,
                 background=(1.0, 1.0, 1.0)):
        self.viewer = viewer
        self.assets = assets or TrialAssets()
        self.bodies = {}
        self.robot = None
        viewer["/Background"].set_property("top_color", list(background))
        viewer["/Background"].set_property("bottom_color", list(background))
        viewer["/Grid"].set_transform(np.eye(4))
        if show_cameras:
            self._draw_cameras()
        self._draw_force_plates()
        if self.assets.table is not None:
            self._draw_table(self.assets.table, table_rgba)
        if self.assets.robot_base is not None:
            self._load_robot(robot_rgba)

    # -- static elements ---------------------------------------------------

    def _draw_cameras(self, size=0.1):
        import meshcat.geometry as g
        for k, T in self.assets.cameras.items():
            node = self.viewer["scene/cameras"][f"camera_{k}"]
            node["body"].set_object(g.Box([size, size, size]), _material((0.01, 0.01, 0.01, 1.0)))
            node.set_transform(T)
        for a, b in CAMERA_SUPPORTS:           # the bar each pair is mounted on
            if a in self.assets.cameras and b in self.assets.cameras:
                p, q = self.assets.cameras[a][:3, 3], self.assets.cameras[b][:3, 3]
                self._bar(f"scene/cameras/support_{a}_{b}", p, q, 0.05, (0.01, 0.01, 0.01, 0.9))

    def _bar(self, path, p, q, thickness, rgba):
        import meshcat.geometry as g
        v = np.asarray(q, float) - np.asarray(p, float)
        length = float(np.linalg.norm(v))
        x = v / max(length, 1e-9)
        up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        y = np.cross(up, x)
        y /= np.linalg.norm(y)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
        T[:3, 3] = 0.5 * (np.asarray(p) + np.asarray(q))
        self.viewer[path].set_object(g.Box([length, thickness, thickness]), _material(rgba))
        self.viewer[path].set_transform(T)

    def _draw_table(self, table, rgba=None):
        """comfi-examples' draw_table, for any size and height: a top and four legs."""
        import meshcat.geometry as g
        top_rgba, leg_rgba = (rgba, rgba) if rgba is not None else (TABLE_TOP_RGBA, TABLE_LEG_RGBA)
        (length, width), height = table["size"], table["height"]
        node = self.viewer["scene/table"]
        node["top"].set_object(g.Box([length, width, TABLE_THICKNESS]), _material(top_rgba))
        top = np.eye(4)
        top[2, 3] = height - TABLE_THICKNESS / 2
        node["top"].set_transform(top)
        xs = (length / 2 - TABLE_INSET - TABLE_LEG / 2, -length / 2 + TABLE_INSET + TABLE_LEG / 2)
        ys = (width / 2 - TABLE_INSET - TABLE_LEG / 2, -width / 2 + TABLE_INSET + TABLE_LEG / 2)
        for i, (x, y) in enumerate((x, y) for x in xs for y in ys):
            leg = np.eye(4)
            leg[:3, 3] = (x, y, (height - TABLE_THICKNESS) / 2)
            node[f"leg_{i}"].set_object(g.Box([TABLE_LEG, TABLE_LEG, height - TABLE_THICKNESS]),
                                        _material(leg_rgba))
            node[f"leg_{i}"].set_transform(leg)
        node.set_transform(table["pose"])

    def _draw_force_plates(self):
        import meshcat.geometry as g
        for i, ((sx, sy), (cx, cy)) in enumerate(self.assets.force_plates, start=1):
            node = self.viewer["scene/force_plates"][f"fp{i}"]
            node.set_object(g.Box([sx, sy, FORCE_PLATE_THICKNESS]), _material(FORCE_PLATE_RGBA))
            T = np.eye(4)
            T[:3, 3] = (cx, cy, FORCE_PLATE_THICKNESS / 2)
            node.set_transform(T)

    def _load_robot(self, rgba=None):
        import example_robot_data as erd
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
        panda = erd.load("panda")
        if rgba is not None:
            _recolour(panda.visual_model, rgba)
        viz = MeshcatVisualizer(panda.model, None, panda.visual_model)
        viz.initViewer(viewer=self.viewer)
        viz.loadViewerModel(rootNodeName="scene/robot", visual_color=rgba)
        self.viewer["scene/robot"].set_transform(self.assets.robot_base)
        self.robot = viz
        self._robot_q = pin.neutral(panda.model)
        self.robot.display(self._robot_q)

    # -- bodies and markers ------------------------------------------------

    def add_body(self, name, model, visual_model, rgba=None):
        """Load a human model under ``bodies/<name>``; ``rgba`` recolours it
        uniformly (alpha < 1 makes it translucent), None keeps its own colours."""
        from pinocchio.visualize import MeshcatVisualizer
        if name in self.bodies:
            self.viewer[f"bodies/{name}"].delete()
        visual_model = visual_model.copy()
        if rgba is not None:
            _recolour(visual_model, rgba)
        viz = MeshcatVisualizer(model, None, visual_model)
        viz.initViewer(viewer=self.viewer)
        viz.loadViewerModel(rootNodeName=f"bodies/{name}", visual_color=rgba)
        self.bodies[name] = viz
        return viz

    def set_markers(self, name, positions, rgba=(0.5, 0.5, 0.5, 1.0), radius=0.012, spheres=True):
        """Draw a marker set under ``markers/<name>``: (n, 3) world positions,
        non-finite rows hidden. Spheres look better; a point cloud is cheaper to
        update every frame."""
        import meshcat.geometry as g
        import meshcat.transformations as tf
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        node = self.viewer["markers"][name]
        if not spheres:
            ok = np.isfinite(positions).all(axis=1)
            colours = np.tile(np.asarray(rgba[:3], np.float32)[:, None], (1, int(ok.sum())))
            node.set_object(g.PointCloud(position=positions[ok].T.astype(np.float32),
                                         color=colours, size=2 * radius))
            return
        node.delete()
        sphere, material = g.Sphere(radius), _material(rgba)
        for i, p in enumerate(positions):
            if np.isfinite(p).all():
                node[str(i)].set_object(sphere, material)
                node[str(i)].set_transform(tf.translation_matrix(p))

    def show(self, frame=None, configurations=None):
        """Pose the bodies (name -> q) and put the robot in its state at video
        ``frame``; the robot keeps its last state on frames without one."""
        for name, q in (configurations or {}).items():
            self.bodies[name].display(np.asarray(q, dtype=float))
        if self.robot is not None and frame is not None:
            joints = self.assets.robot_joints.get(int(frame))
            if joints is not None:
                self._robot_q[:7] = joints
                self.robot.display(self._robot_q)
