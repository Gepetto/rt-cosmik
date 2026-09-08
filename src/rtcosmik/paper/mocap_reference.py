"""Drive the pipeline's own IK from COMFI's mocap markers.

Both arms of the comparison are scored against COMFI's published joint angles,
which were produced by COMFI's model and its inverse kinematics -- not ours. So
every reported error mixes two things: how well the pose estimator recovered the
markers, and how differently the two biomechanical models interpret the same
markers. There is no way to separate them from the published numbers alone.

Running the mocap markers through *our* model and *our* solver separates them.
The result serves twice:

*As a check.* Comparing it with COMFI's published angles for the same trial says
how much of the reported error is model difference rather than estimation error.

*As a reference.* Scoring both arms against it removes the model difference
entirely, leaving only what the pose estimators actually did. It is also the
floor: what this model and solver achieve with markers that are as good as the
dataset gets, which no estimator driving them can beat.

The marker sets almost line up already -- COMFI names its mocap markers the way
RT-COSMIK does -- with one gap. Mocap carries a Vicon head cluster instead of
facial landmarks, so the head comes from that cluster and the three facial
markers that have no counterpart are dropped, which is what ``marker_set =
"mocap"`` means. The head segment only needs Head, REar and LEar, so the
cervical DoF stay observable and the locked set is the same seven as parity.
"""

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: The four Vicon head-cluster markers, on a band around the head.
HEAD_BAND = ("FHD", "BHD", "LHD", "RHD")

#: Cluster markers that stand in directly for a facial landmark. RHD and LHD sit
#: on the right and left of the band, at roughly ear height, so the lateral axis
#: the head segment builds from ``REar - LEar`` is the one it wants.
HEAD_DIRECT = {"REar": "RHD", "LEar": "LHD"}

#: ``Head`` is not one of them. The head segment takes its vertical axis from
#: ``Head - shoulder_centre``, so Head has to sit on the head's vertical axis --
#: and every band marker is off it, FHD by the depth of the forehead. Using FHD
#: tilts that axis forward by 14 degrees on average (15.8 worst case on
#: 1012/Lifting), which lands directly on the
#: cervical angles this reference is supposed to arbitrate. The band's centroid
#: is on the axis, so it is what Head is derived from.
HEAD_DERIVED = "Head"

MM_TO_M = 1e-3


def load_mocap_markers(mocap_dir, marker_names):
    """Read one trial's mocap markers, named and scaled as the pipeline wants.

    COMFI writes ``<NAME>_X[mm]`` columns in millimetres; the pipeline works in
    metres with bare marker names.

    Returns:
        (frames, names, array): the marker names actually found, and an
        ``(F, M, 3)`` array in metres.
    """
    import csv

    path = Path(mocap_dir) / "markers_trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(f"no mocap markers at {path}")
    with open(path) as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    index = {name: i for i, name in enumerate(header)}

    def columns_for(source):
        axes = [index.get(f"{source}_{a}[mm]") for a in "XYZ"]
        return None if any(a is None for a in axes) else axes

    band = {name: columns_for(name) for name in HEAD_BAND}
    have_band = all(v is not None for v in band.values())

    wanted, columns, derived = [], [], []
    for name in marker_names:
        if name == HEAD_DERIVED:
            if have_band:
                wanted.append(name)
                columns.append(None)
                derived.append(len(wanted) - 1)
            continue
        axes = columns_for(HEAD_DIRECT.get(name, name))
        if axes is None:
            continue
        wanted.append(name)
        columns.append(axes)

    def value(row, col):
        raw = row[col]
        return float(raw) * MM_TO_M if raw not in ("", "nan") else np.nan

    data = np.empty((len(body), len(wanted), 3), dtype=np.float64)
    for f, row in enumerate(body):
        for m, axes in enumerate(columns):
            if axes is None:
                continue
            data[f, m, :] = [value(row, c) for c in axes]
        for m in derived:
            data[f, m, :] = np.mean(
                [[value(row, c) for c in band[n]] for n in HEAD_BAND], axis=0)
    return len(body), wanted, data


class MocapMarkerSource:
    """Yield ``(frame, mks_dict)`` from a mocap trial, like the estimator arms.

    Deliberately no filtering: the arms filter because triangulated markers are
    noisy, and mocap is the thing that noise is measured against. Smoothing it
    would make the reference something other than what the dataset recorded.
    """

    def __init__(self, mocap_dir, marker_names, logger=None):
        self.frames, self.names, self.data = load_mocap_markers(
            mocap_dir, marker_names)
        self.logger = logger or LOGGER
        missing = [n for n in marker_names if n not in self.names]
        if missing:
            self.logger.warning(
                f"mocap trial {mocap_dir} lacks {len(missing)} of the "
                f"configured markers: {', '.join(missing)}")

    def __len__(self):
        return self.frames

    def __iter__(self):
        for frame in range(self.frames):
            row = self.data[frame]
            if not np.isfinite(row).all():
                # A dropped mocap marker cannot be interpolated away here
                # without inventing data; skip the frame and let the frame
                # counter record the gap.
                continue
            yield frame, dict(zip(self.names, row))
