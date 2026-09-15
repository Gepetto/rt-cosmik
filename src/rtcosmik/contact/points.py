"""Points foot contact reads and holds, and how firmly."""

import numpy as np

#: ContactVision output order: left toe, right toe, left heel, right heel.
PROBABILITY_MARKERS = ("LTOE", "RTOE", "LHEE", "RHEE")

#: Marker frames held by the MHE -> probability they follow.
CONTACT_POINTS = {"RHEE": "RHEE", "R5MHD": "RTOE", "RTOE": "RTOE",
                  "LHEE": "LHEE", "L5MHD": "LTOE", "LTOE": "LTOE"}

# Holding weights at probability 1, and anchor follow rate per frame
# (tuned on 11 COMFI trials: legs never worse than +0.1 deg).
W_SLIP = 1e-2
W_ANCHOR = 0.3
ANCHOR_FOLLOW = 0.05

#: ContactVision input order (OpenPose BODY_25 lower body).
LOWER_BODY = ("MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
              "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel")

#: OpenPose foot points that are COSMIK markers: (marker, SMPL-X vertex).
FOOT_MARKERS = {"LBigToe": ("LTOE", 5770), "LSmallToe": ("L5MHD", 5780),
                "LHeel": ("LHEE", 8846), "RBigToe": ("RTOE", 8463),
                "RSmallToe": ("R5MHD", 8474), "RHeel": ("RHEE", 8635)}

#: The other points: SMPL-X joints 0, 2, 5, 8, 1, 4, 7 in NLF canonical space.
JOINT_CANONICAL = {
    "MidHip": (0.001105, -0.206752, 0.037434),
    "RHip": (-0.062748, -0.309953, 0.016787),
    "RKnee": (-0.32432, -0.568062, 0.004138),
    "RAnkle": (-0.579692, -0.893272, -0.016593),
    "LHip": (0.059261, -0.300374, 0.012204),
    "LKnee": (0.337752, -0.565604, 0.00664),
    "LAnkle": (0.564945, -0.903682, -0.026215),
}


def extra_canonical_points():
    """Canonical points appended to NLF's query, (7, 3)."""
    return np.array(list(JOINT_CANONICAL.values()), dtype=np.float32)


def lower_body_indices(marker_names, nlf_indices):
    """Index of each LOWER_BODY point in NLF's output (markers, then extra points)."""
    marker_names = list(marker_names)
    extra = list(JOINT_CANONICAL)
    indices = []
    for name in LOWER_BODY:
        if name in JOINT_CANONICAL:
            indices.append(len(marker_names) + extra.index(name))
            continue
        marker, vertex = FOOT_MARKERS[name]
        if marker not in marker_names:
            raise ValueError(f"foot contact needs the {marker} marker ({name})")
        position = marker_names.index(marker)
        if int(nlf_indices[position]) != vertex:
            raise ValueError(
                f"{marker} queries SMPL-X vertex {nlf_indices[position]}, but foot "
                f"contact expects vertex {vertex} (OpenPose {name})")
        indices.append(position)
    return np.array(indices)
