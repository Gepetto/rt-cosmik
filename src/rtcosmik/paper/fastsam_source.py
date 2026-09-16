"""Drive the pipeline's IK from COMFI's FastSAM-3D marker export.

This is the third estimation modality in the study, and the second that
regresses metric 3D directly rather than triangulating image landmarks. FastSAM
was run once per camera, and each run is exported as its own CSV of 36 named
markers per frame, in metres, **in that camera's own frame**::

    <dataset>/fastsam/<participant>/<task>/cosmik_mhr_markers_cam{0,2,4,6}.csv

So there is no detector to run here, but there is a reconstruction, and it is
the NLF arm's, step for step: each view stays in its camera's frame, the views
are fused into the reference camera's frame by the same
:func:`fuse_camera_poses3d`, the result is mapped into the world with the
reference camera's pose, and it is low-passed one frame at a time. The one
unavoidable difference is the fusion weights: NLF reports a per-point
uncertainty and its views are combined by inverse variance, while FastSAM
reports none, so its views are averaged with equal weight.

A single camera goes through the same fusion. The reference camera's projection
is the identity, so one view comes out unchanged, and ``cameras=[0]`` reproduces
the earlier single-camera arm.

The marker set is a near-exact match for parity, which is what makes the
comparison fair without any tuning:

* 34 of the 35 parity markers are present under identical names.
* ``TV8`` and ``TV12`` are extra, and are dropped. Parity already drops the
  thoracic markers (``T11``, ``T6``) and locks the three ``middle_thoracic``
  DoF, so keeping them would give this arm an observable joint the other arms
  do not have.
* ``Head`` is missing, and cannot simply be dropped -- see below.

``Head`` matters more than its single column suggests. ``get_head_pose`` builds
the head segment's vertical axis from ``Head - shoulder_centre``, and
``construct_segments_frames`` only adds the head segment when Head, REar and
LEar are all present. Without it the head frame is never built, the six facial
markers are never registered, and this arm would be solving a structurally
different model from every other arm.

So Head is reconstructed from the facial landmarks FastSAM does export. Where to
put it was measured rather than guessed: over all 108 NLF single-camera runs,
``Head`` sits at a stable offset from the ear midpoint when expressed in a
head-local frame built from the ears and the nose and scaled by ear width
(:data:`HEAD_OFFSET`, spread 0.03-0.05 ear widths, i.e. 5-8 mm). FastSAM's own
faces are the same size as NLF's -- ear width 146-158 mm against NLF's 150 mm
mean -- so the offset transfers without rescaling.

Taking the offset from NLF rather than from mocap is deliberate. Mocap carries a
Vicon head band, not facial landmarks, and it is the reference this arm is
scored against; deriving a marker from the reference would tune the estimate
toward its own ground truth. NLF is an independent modality, is the arm this one
is most directly compared with, and places Head three to five times more
repeatably than the 2D arm does.

The result does not rest on that choice. Re-solving six trials with alternative
placements moved the whole-body joint RMSE by +0.003 deg (the NLF 4-camera
offset), +0.006 deg (a 10 percent shorter head), +0.109 deg (straight up, no
forward or lateral tilt) and +0.178 deg (a deliberately wrong 20 deg forward
tilt) -- against a roughly 2 deg spread between the arms being compared. See
``scripts/python/paper/study_head_offset.py``.
"""

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: One CSV per camera, under ``<dataset>/fastsam/<participant>/<task>/``.
FASTSAM_FILE = "cosmik_mhr_markers_cam{camera}.csv"

#: Participants left out of every FastSAM arm. 3361's results come from a
#: different FastSAM inference script and do not line up with COMFI's
#: calibration: its camera-0 range is scaled by about 2, and cameras 2/4/6 land
#: 0.5-0.9 m from the mocap markers whichever camera pose is used. Being
#: investigated upstream; until then it is reported as excluded, not run.
EXCLUDED_PARTICIPANTS = ("3361",)

#: Columns that carry no marker.
META_COLUMNS = ("frame_id", "person_id", "valid")

#: Exported markers with no counterpart in the parity set. Parity drops the
#: thoracic markers and locks the middle_thoracic DoF, so these would add an
#: observability the other arms do not have.
DROPPED_MARKERS = ("TV8", "TV12")

#: ``Head`` in the ear/nose frame, in units of ear width: (forward, up,
#: lateral). Measured as the mean over all 108 ``nlf_0`` runs; the per-trial
#: spread is (0.053, 0.031, 0.038), so about (8, 5, 6) mm on a 150 mm face.
HEAD_OFFSET = (-0.216, 0.881, -0.058)

#: Facial landmarks the offset is anchored to.
HEAD_ANCHORS = ("REar", "LEar", "Nose")


def fastsam_csv(trial_dir, camera):
    """Path of one camera's export for one trial."""
    return Path(trial_dir) / FASTSAM_FILE.format(camera=camera)


def load_fastsam_markers(path):
    """Read one camera's export. Returns ``(names, xyz, valid)``.

    ``xyz`` is ``(frames, markers, 3)`` in metres, in that camera's own frame.
    ``valid`` is the exporter's per-frame flag; invalid frames hold NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no FastSAM export at {path}")
    with open(path) as handle:
        header = handle.readline().strip().split(",")
    if tuple(header[:3]) != META_COLUMNS:
        raise ValueError(f"unexpected FastSAM header in {path}: {header[:3]}")

    # Columns run <NAME>_X[m], <NAME>_Y[m], <NAME>_Z[m] per marker, in order.
    names = [column[: -len("_X[m]")] for column in header[3::3]]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None]
    xyz = data[:, 3:].reshape(len(data), len(names), 3)
    return names, xyz, data[:, 2].astype(int)


def derive_head(markers):
    """Place ``Head`` from the facial landmarks, in whatever frame they are in.

    ``markers`` maps names to ``(3,)`` arrays. The construction is a rotation of
    the anchors plus a scaling by ear width, so it commutes with the rigid
    camera-to-world transform and gives the same answer either side of it.
    """
    right, left, nose = (np.asarray(markers[name], dtype=float)
                         for name in HEAD_ANCHORS)
    midpoint = (right + left) / 2.0

    lateral = right - left
    width = np.linalg.norm(lateral)
    if width < 1e-6:
        raise ValueError("degenerate ear width; cannot place Head")
    lateral = lateral / width

    forward = nose - midpoint
    forward = forward - forward.dot(lateral) * lateral
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise ValueError("nose on the ear axis; cannot place Head")
    forward = forward / norm

    up = np.cross(lateral, forward)
    a, b, c = HEAD_OFFSET
    return midpoint + width * (a * forward + b * up + c * lateral)


class FastsamMarkerSource:
    """Iterate one trial as ``(frame, {marker: xyz})`` in the world frame.

    Mirrors the NLF arm's offline path exactly: per-view markers in each
    camera's frame, fused into the reference camera's frame, mapped into the
    world with the reference camera's pose, then low-passed one frame at a time.
    """

    def __init__(self, trial_dir, cameras, marker_names, projections,
                 world_R, world_T, iir=None):
        from rtcosmik.triangulation.triangulation import fuse_camera_poses3d

        self._fuse = fuse_camera_poses3d
        self.cameras = list(cameras)
        self.marker_names = list(marker_names)
        self.projections = projections
        self.world_R = np.asarray(world_R, dtype=float)
        self.world_T = np.asarray(world_T, dtype=float)
        self.iir = iir

        self.views = []
        for camera in self.cameras:
            path = fastsam_csv(trial_dir, camera)
            names, xyz, valid = load_fastsam_markers(path)
            keep = [name for name in names if name not in DROPPED_MARKERS]
            missing = (set(self.marker_names) - set(keep)) - {"Head"}
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            index = {name: i for i, name in enumerate(names)}
            columns = {name: index[name] for name in keep}
            needs_head = "Head" in self.marker_names and "Head" not in index
            self.views.append((columns, xyz, valid.astype(bool), needs_head))

        # The cameras can disagree on length by a frame. Like the NLF arm's video
        # reader, which stops when any stream ends, keep only frames every view has.
        self.length = min(len(xyz) for _, xyz, _, _ in self.views)

    def __len__(self):
        return self.length

    def view_pose(self, view, frame):
        """One camera's parity markers for one frame, in its own frame, or None."""
        columns, xyz, valid, needs_head = self.views[view]
        if not valid[frame]:
            return None
        markers = {name: xyz[frame, column] for name, column in columns.items()}
        if needs_head:
            markers["Head"] = derive_head(markers)
        pose = np.stack([markers[name] for name in self.marker_names])
        return pose if np.isfinite(pose).all() else None

    def world_markers(self, frame):
        """One frame's fused parity markers in the world frame, unfiltered, or None."""
        poses = [self.view_pose(view, frame) for view in range(len(self.views))]
        fused = self._fuse(poses, self.projections)
        if len(fused) == 0:
            return None
        return fused @ self.world_R.T + self.world_T

    def __iter__(self):
        for frame in range(len(self)):
            points = self.world_markers(frame)
            if points is None:
                continue
            if self.iir is not None:
                points = self.iir(points)
            yield frame, dict(zip(self.marker_names, points))


def build_source(dataset, participant, task, cameras, settings, logger=None):
    """Assemble the marker source for one trial, with its subject metadata.

    Calibration is loaded exactly as the NLF arm loads it: projections for the
    requested cameras, and the world pose of the first one, which is the
    reference frame the views are fused into.
    """
    import yaml

    from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                           load_world_transformation)
    from rtcosmik.filtering.iir import MarkerFilter

    if participant in EXCLUDED_PARTICIPANTS:
        raise ValueError(f"participant {participant} is excluded from the FastSAM "
                         f"arms (see EXCLUDED_PARTICIPANTS)")

    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    _, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = load_world_transformation(cam_dir, cameras[0])

    iir = MarkerFilter(len(settings.marker_names), settings)

    source = FastsamMarkerSource(
        root / "fastsam" / participant / task, cameras, settings.marker_names,
        projections, world_R, world_T, iir=iir)
    (logger or LOGGER).info(
        f"FastSAM {participant}/{task}: {len(source)} frames, cameras "
        f"{list(cameras)}, {len(settings.marker_names)} markers")
    return source, meta
