"""Marker sets and the paper-evaluation profile, applied to a settings object."""

#: "parity": the 35 markers the RTMPose+LSTM baseline also produces, with the
#: 7 DoF it cannot observe locked. "mocap" also drops the facial landmarks the
#: mocap reference lacks. "nlf": every marker, nothing locked.
PARITY_DROPPED_MARKERS = ("T11", "T6", "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN")
PARITY_LOCKED_JOINTS = ("middle_thoracic_Z", "middle_thoracic_X", "middle_thoracic_Y",
                        "left_wrist_Z", "left_wrist_X", "right_wrist_Z", "right_wrist_X")
MOCAP_DROPPED_MARKERS = ("Nose", "REye", "LEye")
#: OpenCap's mocap has no head markers, and medial knee and ankle markers in the
#: static trial only: "opencap_mocap" drops them from "mocap" and locks the
#: cervical DoF; "opencap_parity" is "parity" without the head.
HEAD_MARKERS = ("Nose", "Head", "REar", "LEar", "REye", "LEye")
MEDIAL_LEG_MARKERS = ("RMKNE", "LMKNE", "RMANK", "LMANK")
CERVICAL_JOINTS = ("middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y")
#: marker set -> (markers dropped from the full list, joints locked)
MARKER_SETS = {
    "parity": (PARITY_DROPPED_MARKERS, PARITY_LOCKED_JOINTS),
    "mocap": (PARITY_DROPPED_MARKERS + MOCAP_DROPPED_MARKERS, PARITY_LOCKED_JOINTS),
    "opencap_mocap": (PARITY_DROPPED_MARKERS + HEAD_MARKERS + MEDIAL_LEG_MARKERS,
                      PARITY_LOCKED_JOINTS + CERVICAL_JOINTS),
    "opencap_parity": (PARITY_DROPPED_MARKERS + HEAD_MARKERS,
                       PARITY_LOCKED_JOINTS + CERVICAL_JOINTS),
}


def apply_marker_set(target, name):
    """Restrict ``target`` to one marker set; can be re-applied."""
    if not hasattr(target, "full_marker_names"):
        target.full_marker_names = list(target.marker_names)
        target.full_keys_to_track = list(target.keys_to_track_list)
        target.full_nlf_indices = list(target.nlf_indices)
    target.marker_names = list(target.full_marker_names)
    target.keys_to_track_list = list(target.full_keys_to_track)
    target.nlf_indices = list(target.full_nlf_indices)
    target.marker_set = name
    if name == "nlf":
        target.locked_joints = []
        return target
    if name not in MARKER_SETS:
        raise ValueError(f"unknown marker set {name!r}")
    dropped, locked = MARKER_SETS[name]
    keep = [i for i, m in enumerate(target.marker_names) if m not in dropped]
    target.nlf_indices = [target.nlf_indices[i] for i in keep]
    target.marker_names = [target.marker_names[i] for i in keep]
    target.keys_to_track_list = [k for k in target.keys_to_track_list if k not in dropped]
    target.locked_joints = list(locked)
    return target


def apply_evaluation(target):
    """Apply ``target.paper_evaluation``: the paper's filter, horizon, vertices and markers."""
    if not hasattr(target, "toolbox_nlf_indices"):
        target.toolbox_nlf_indices = list(target.nlf_indices)
    if target.paper_evaluation:
        p = target.paper
        target.order, target.cutoff_freq, target.N = p["order"], p["cutoff_freq"], p["N"]
        target.filter_type = p["filter_type"]
        target.one_euro_min_cutoff, target.one_euro_beta = p["one_euro_min_cutoff"], p["one_euro_beta"]
        target.nlf_indices = list(p["nlf_indices"])
        target.marker_set = p["marker_set"]
    return apply_marker_set(target, target.marker_set)
