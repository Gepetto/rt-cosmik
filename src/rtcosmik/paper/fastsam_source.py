"""Drive the pipeline's IK from COMFI's FastSAM-3D marker export.

This is the third estimation modality in the study, and the second that
regresses metric 3D directly rather than triangulating image landmarks. COMFI
ships it as one CSV per trial holding 36 named markers per frame, already in
metres and **already in the reference camera's frame** -- the ``_cam`` in the
file name. So unlike the other arms there is no detector to run and no
reconstruction to choose: the only thing between the file and the solver is the
camera-to-world anchor, the same ``p_world = R p_cam + T`` every arm applies to
whatever it reconstructs.

Only camera 0 is exported for now, so this arm is single-view by construction
and its natural comparison is ``nlf_0``.

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
from collections import deque
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: One CSV per trial, under ``<dataset>/fastsam/<participant>/<task>/``.
FASTSAM_FILE = "cosmik_mhr_markers_cam.csv"

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


def load_fastsam_markers(trial_dir):
    """Read one trial's export. Returns ``(names, xyz, valid)``.

    ``xyz`` is ``(frames, markers, 3)`` in metres, in the reference camera's
    frame. ``valid`` is the exporter's own per-frame flag.
    """
    path = Path(trial_dir) / FASTSAM_FILE
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

    Mirrors the NLF arm's offline path exactly: transform into the world frame,
    hold a buffer of ``N`` frames seeded from the first sample so filtering
    starts immediately rather than after a silent warm-up, low-pass the buffer,
    and keep its last sample.
    """

    def __init__(self, trial_dir, marker_names, world_R, world_T,
                 iir=None, buffer_len=1):
        self.marker_names = list(marker_names)
        self.world_R = np.asarray(world_R, dtype=float)
        self.world_T = np.asarray(world_T, dtype=float)
        self.iir = iir
        self.buffer_len = buffer_len

        names, xyz, valid = load_fastsam_markers(trial_dir)
        self.valid = valid
        keep = [name for name in names if name not in DROPPED_MARKERS]
        missing = (set(self.marker_names) - set(keep)) - {"Head"}
        if missing:
            raise ValueError(f"FastSAM export is missing {sorted(missing)}")
        index = {name: i for i, name in enumerate(names)}
        self.columns = {name: index[name] for name in keep}
        self.xyz = xyz
        self.needs_head = "Head" in self.marker_names and "Head" not in index

    def __len__(self):
        return len(self.xyz)

    def world_markers(self, frame):
        """One frame's parity markers, in the world frame, unfiltered."""
        points = self.xyz[frame] @ self.world_R.T + self.world_T
        markers = {name: points[column] for name, column in self.columns.items()}
        if self.needs_head:
            markers["Head"] = derive_head(markers)
        return markers

    def __iter__(self):
        channels = 3 * len(self.marker_names)
        buffer = deque(maxlen=self.buffer_len)
        for frame in range(len(self)):
            if not self.valid[frame]:
                continue
            markers = self.world_markers(frame)
            stacked = np.stack([markers[name] for name in self.marker_names])

            if not buffer:
                for _ in range(self.buffer_len):
                    buffer.append(stacked)
            else:
                buffer.append(stacked)
            if self.iir is not None:
                block = np.asarray(buffer).reshape(self.buffer_len, channels)
                stacked = self.iir.filter(block).reshape(
                    self.buffer_len, len(self.marker_names), 3)[-1]
            yield frame, dict(zip(self.marker_names, stacked))


def build_source(dataset, participant, task, cameras, settings, logger=None):
    """Assemble the marker source for one trial, with its subject metadata.

    ``cameras`` selects the world anchor. FastSAM is exported from camera 0
    only, so anything else is refused rather than silently anchored wrong.
    """
    import yaml

    from rtcosmik.camera.cam_utils import load_world_transformation
    from rtcosmik.filtering.iir import IIR

    if tuple(cameras) != (0,):
        raise ValueError(
            f"FastSAM is exported from camera 0 only, got cameras={list(cameras)}")

    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    world_R, world_T = load_world_transformation(
        root / "cam_params" / participant, cameras[0])

    iir = IIR(num_channel=3 * len(settings.marker_names),
              sampling_frequency=settings.fs)
    iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                   filter_type=settings.filter_type)

    source = FastsamMarkerSource(
        root / "fastsam" / participant / task, settings.marker_names,
        world_R, world_T, iir=iir, buffer_len=settings.N)
    (logger or LOGGER).info(
        f"FastSAM {participant}/{task}: {len(source)} frames, camera "
        f"{cameras[0]}, {len(settings.marker_names)} markers")
    return source, meta
