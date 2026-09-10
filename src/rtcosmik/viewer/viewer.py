"""3D Visualization backends for calibrated human model display.

Supports three backend options:
  - "viser"   (default): Viser web visualizer with custom geometry mapping via ViserRobotVisualizer.
  - "meshcat": Meshcat web visualizer using Pinocchio's MeshcatVisualizer directly.
  - "none"   : No-op fallback for headless execution.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pinocchio as pin

LOGGER = logging.getLogger(__name__)


class ViserRobotVisualizer:
    """Pinocchio geometry loader and renderer for Viser."""

    def __init__(self, model: pin.Model, collision_model: pin.GeometryModel, visual_model: pin.GeometryModel):
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.data = model.createData()
        self.visual_data = visual_model.createData()
        self.server = None
        self.root = "ref"
        self._handles = []

    def initViewer(self, viewer):
        self.server = viewer

    def loadViewerModel(self, rootNodeName="ref"):
        self.root = rootNodeName
        self._handles = []
        for i, geom_obj in enumerate(self.visual_model.geometryObjects):
            try:
                mesh = self._geom_to_trimesh(geom_obj)
            except Exception as exc:
                LOGGER.warning(f"ViserRobotVisualizer: skipping '{geom_obj.name}' ({exc})")
                self._handles.append(None)
                continue
            path = f"/{self.root}/{i:03d}_{geom_obj.name}"
            handle = self.server.scene.add_mesh_trimesh(path, mesh)
            self._handles.append(handle)

    def _geom_to_trimesh(self, geom_obj):
        import trimesh

        geom = geom_obj.geometry
        gtype = type(geom).__name__

        if gtype == "Capsule":
            mesh = trimesh.creation.capsule(
                radius=geom.radius, height=2.0 * geom.halfLength, count=(8, 8)
            )
        elif gtype == "Sphere":
            mesh = trimesh.creation.icosphere(radius=geom.radius, subdivisions=2)
        elif gtype == "Cylinder":
            mesh = trimesh.creation.cylinder(
                radius=geom.radius, height=2.0 * geom.halfLength, sections=16
            )
        elif gtype == "Box":
            mesh = trimesh.creation.box(extents=2.0 * np.array(geom.halfSide))
        else:
            if getattr(geom_obj, "meshPath", None):
                mesh = trimesh.load(geom_obj.meshPath, force="mesh")
            else:
                raise ValueError(
                    f"Unsupported geometry type '{gtype}' for '{geom_obj.name}' "
                    "and no meshPath to fall back on."
                )

        color = getattr(geom_obj, "meshColor", None)
        if color is not None and len(color) >= 3:
            rgba = np.array(
                [color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0]
            )
            mesh.visual.vertex_colors = np.tile(
                (rgba * 255).astype(np.uint8), (len(mesh.vertices), 1)
            )

        scale = np.asarray(getattr(geom_obj, "meshScale", [1.0, 1.0, 1.0]), dtype=float).flatten()
        if not np.allclose(scale, 1.0):
            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = scale
            mesh.apply_transform(S)

        return mesh

    def display(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
        for i, handle in enumerate(self._handles):
            if handle is None:
                continue
            oMg = self.visual_data.oMg[i]
            quat = pin.Quaternion(oMg.rotation)
            handle.wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
            handle.position = np.asarray(oMg.translation, dtype=np.float64)

    def displayCollisions(self, flag: bool):
        pass

    def displayVisuals(self, flag: bool):
        pass


class ViserViewer:
    """3D display backend using Viser."""

    def __init__(self, model, collision_model, visual_model, marker_names, freeflyer=True):
        import viser

        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.marker_names = marker_names
        self.freeflyer = freeflyer

        self.server = viser.ViserServer()
        LOGGER.info(f"[INFO] Viser visualizer available here: http://{self.server.get_host()}:{self.server.get_port()}")

        self.server.scene.add_grid(
            "/grid",
            width=10.0,
            height=10.0,
            position=(0.0, 0.0, 0.0),
        )

        self.marker_colors = np.zeros((len(self.marker_names), 3), dtype=np.uint8)
        self.marker_colors[:, 0] = 255  # Red

        self._markers_handle = None

        self.viz_human = ViserRobotVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz_human.initViewer(viewer=self.server)
        self.viz_human.loadViewerModel(rootNodeName="ref")

        try:
            self.server.scene.set_background_image(None)
        except Exception:
            pass

    def display_q(self, q):
        self.viz_human.display(q)

    def display_markers(self, pos_markers_dict: Dict[str, np.ndarray]):
        pts = np.stack(list(pos_markers_dict.values()), axis=0).astype(np.float32)
        self._markers_handle = self.server.scene.add_point_cloud(
            name="/markers",
            points=pts,
            colors=self.marker_colors,
            point_size=0.02,
        )


class MeshcatViewer:
    """3D display backend using Meshcat."""

    def __init__(self, model, collision_model, visual_model, marker_names, freeflyer=True):
        import meshcat
        import meshcat.geometry as g
        from pinocchio.visualize import MeshcatVisualizer

        self._g = g
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.marker_names = marker_names
        self.freeflyer = freeflyer

        self.vis = meshcat.Visualizer()
        LOGGER.info(f"[INFO] Meshcat visualizer available here: {self.vis.url()}")
        self.vis_markers = self.vis["markers"]

        self.marker_colors = np.zeros((3, len(self.marker_names)), dtype=np.float32)
        self.marker_colors[0, :] = 1.0  # Red

        self.viz_human = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz_human.initViewer(self.vis, open=False)

        try:
            self.vis["ref"].delete()
        except Exception:
            pass
        self.viz_human.loadViewerModel("ref")

        self.viz_human.viewer["/Background"].set_property("top_color", [1, 1, 1])
        self.viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])

    def display_q(self, q):
        self.viz_human.display(q)

    def display_markers(self, pos_markers_dict: Dict[str, np.ndarray]):
        pts = np.stack(list(pos_markers_dict.values()), axis=0).astype(np.float32)
        self.vis_markers.set_object(
            self._g.PointCloud(position=pts.T, color=self.marker_colors, size=0.02)
        )


class NoOpViewer:
    """Fallback no-op viewer for headless mode ("none")."""

    def __init__(self, model, collision_model, visual_model, marker_names, freeflyer=True):
        pass

    def display_q(self, q):
        pass

    def display_markers(self, pos_markers_dict):
        pass


class Viewer:
    """Unified Viewer class wrapping Viser, Meshcat, or No-Op backends."""

    def __init__(
        self,
        model: pin.Model,
        collision_model: pin.GeometryModel,
        visual_model: pin.GeometryModel,
        marker_names: List[str],
        freeflyer: bool = True,
        backend: str = "viser",
    ):
        backend = backend.lower()
        if backend == "viser":
            self._impl = ViserViewer(model, collision_model, visual_model, marker_names, freeflyer)
        elif backend == "meshcat":
            self._impl = MeshcatViewer(model, collision_model, visual_model, marker_names, freeflyer)
        elif backend == "none":
            self._impl = NoOpViewer(model, collision_model, visual_model, marker_names, freeflyer)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choice must be 'viser', 'meshcat', or 'none'.")

    def display_q(self, q):
        self._impl.display_q(q)

    def display_markers(self, pos_markers_dict: Dict[str, np.ndarray]):
        self._impl.display_markers(pos_markers_dict)