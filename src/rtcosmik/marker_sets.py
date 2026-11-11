"""Marker sets and the paper-evaluation profile, applied to a settings object."""

#: "parity": the 35 markers the RTMPose+LSTM baseline also produces, with the
#: 7 DoF it cannot observe locked. "mocap" also drops the facial landmarks the
#: mocap reference lacks. "nlf": every marker, nothing locked.
PARITY_DROPPED_MARKERS = ("T11", "T6", "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN")
PARITY_LOCKED_JOINTS = ("middle_thoracic_Z", "middle_thoracic_X", "middle_thoracic_Y",
                        "left_wrist_Z", "left_wrist_X", "right_wrist_Z", "right_wrist_X")
MOCAP_DROPPED_MARKERS = ("Nose", "REye", "LEye")


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
    if name not in ("parity", "mocap"):
        raise ValueError(f"unknown marker set {name!r}")
    dropped = set(PARITY_DROPPED_MARKERS)
    if name == "mocap":
        dropped |= set(MOCAP_DROPPED_MARKERS)
    keep = [i for i, m in enumerate(target.marker_names) if m not in dropped]
    target.nlf_indices = [target.nlf_indices[i] for i in keep]
    target.marker_names = [target.marker_names[i] for i in keep]
    target.keys_to_track_list = [k for k in target.keys_to_track_list if k not in dropped]
    target.locked_joints = list(PARITY_LOCKED_JOINTS)
    return target


def apply_evaluation(target):
    """Apply ``target.paper_evaluation``: the paper's filter, horizon, vertices and markers."""
    if target.paper_evaluation:
        p = target.paper
        target.order, target.cutoff_freq, target.N = p["order"], p["cutoff_freq"], p["N"]
        target.nlf_indices = list(p["nlf_indices"])
        target.marker_set = p["marker_set"]
    return apply_marker_set(target, target.marker_set)
