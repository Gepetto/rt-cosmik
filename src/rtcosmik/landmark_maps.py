"""Landmark maps: the template vertex each landmark is read from, and who it was fitted on."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtcosmik.folds import PARTICIPANTS, fitting_participants, fold_of

MAPS_DIR = Path(__file__).resolve().parents[2] / "config" / "landmark_maps"

#: The submitted NLF and FastSAM-3D maps were both fitted on every participant but 3361.
SUBMITTED_FITTED_ON = tuple(p for p in PARTICIPANTS if p != "3361")


class LeakageError(RuntimeError):
    """The landmark map was fitted on the participant being evaluated."""


@dataclass(frozen=True)
class LandmarkMap:
    front_end: str
    fold: object                      # fold index, or None for the submitted map
    fitted_on: tuple
    vertices: dict                    # landmark -> fitted vertex

    def to_dict(self):
        return {"front_end": self.front_end, "fold": self.fold,
                "fitted_on": list(self.fitted_on), "vertices": dict(self.vertices)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["front_end"], d["fold"], tuple(d["fitted_on"]),
                   {k: int(v) for k, v in d["vertices"].items()})

    def identity(self, participant):
        """What run information records about this map for one participant."""
        digest = hashlib.sha256(json.dumps(sorted(self.vertices.items())).encode()).hexdigest()
        return {"front_end": self.front_end, "fold": self.fold,
                "fitted_on": list(self.fitted_on), "sha256": digest,
                "in_sample": str(participant) in self.fitted_on}


def path(front_end, fold):
    return MAPS_DIR / f"{front_end}_fold{fold}.json"


def load(front_end, fold):
    """Fold ``fold``'s map, as written by the fitting script."""
    m = LandmarkMap.from_dict(json.loads(path(front_end, fold).read_text()))
    if m.front_end != front_end or m.fold != fold or not set(m.fitted_on) <= set(fitting_participants(fold)):
        raise ValueError(f"{path(front_end, fold)} does not match fold {fold}")
    return m


def submitted(settings, front_end="nlf"):
    """The submitted manuscript's map; FastSAM-3D's lives in its exported files."""
    vertices = (dict(zip(settings.full_marker_names, settings.paper["nlf_indices"]))
                if front_end == "nlf" else {})
    return LandmarkMap(front_end, None, SUBMITTED_FITTED_ON, vertices)


def select(settings, participant, front_end="nlf"):
    """The map ``settings.paper["landmark_map"]`` asks for, for one participant."""
    choice = settings.paper["landmark_map"]
    if choice == "submitted":
        return submitted(settings, front_end)
    if choice == "out_of_fold":
        return load(front_end, fold_of(participant))
    raise ValueError(f"unknown landmark_map {choice!r}")


def require_out_of_fold(landmark_map, participant, settings):
    """Refuse a map fitted on ``participant``, unless the submitted map was asked for."""
    if str(participant) not in landmark_map.fitted_on:
        return
    if landmark_map.fold is None and settings.paper["landmark_map"] == "submitted":
        return
    raise LeakageError(f"{landmark_map.front_end} map of fold {landmark_map.fold} was fitted "
                       f"on participant {participant}")


def recorded(views_path, settings):
    """The map a views recording was made with; recordings older than fold maps used the submitted one."""
    data = np.load(views_path)
    if "landmark_map" in data:
        return LandmarkMap.from_dict(json.loads(str(data["landmark_map"])))
    return submitted(settings)


def checked_recording(views_path, participant, settings):
    """Run-information identity of a recording's map, refusing one fitted on ``participant``."""
    landmark_map = recorded(views_path, settings)
    require_out_of_fold(landmark_map, participant, settings)
    return landmark_map.identity(participant)


def nlf_indices(landmark_map, settings):
    """All of NLF's vertices: the map's fitted ones, the hand-picked paper ones elsewhere."""
    return [landmark_map.vertices.get(name, base)
            for name, base in zip(settings.full_marker_names, settings.paper["nlf_indices"])]


def apply(landmark_map, settings):
    """A copy of ``settings`` querying NLF at the map's vertices."""
    import copy
    from rtcosmik.marker_sets import apply_marker_set
    s = copy.copy(settings)
    s.full_nlf_indices = nlf_indices(landmark_map, settings)
    return apply_marker_set(s, s.marker_set)
