import copy
import json

import numpy as np
import pytest

from .conftest import needs_comfi


def _settings(choice="out_of_fold"):
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.ik_type = "sbs"
    s.paper["landmark_map"] = choice
    s.paper["clock_offset"] = 0
    return s


def _stamp(views, fold):
    """Record the recording as made with fold ``fold``'s NLF map."""
    from rtcosmik.folds import fitting_participants
    data = dict(np.load(views))
    data["landmark_map"] = json.dumps({"front_end": "nlf", "fold": fold,
                                       "fitted_on": list(fitting_participants(fold)),
                                       "vertices": {"RKNE": 6403}})
    np.savez_compressed(views, **data)


def _replay(trial, repo, settings):
    from replay_nlf_arms import replay_trial
    arm = {"name": "nlf", "tag": "nlf_0-2-4-6", "reconstruction": "fuse3d",
           "cameras": trial["cameras"]}
    return replay_trial(trial["dataset"], trial["views"], trial["out_dir"], trial["participant"],
                        trial["task"], arm, settings, repo=repo)


@needs_comfi
def test_map_fitted_on_the_participant_is_refused(synthetic_trial, clean_repo):
    from rtcosmik.landmark_maps import LeakageError
    _stamp(synthetic_trial["views"], fold=1)          # fold 1's map was fitted on 1012

    with pytest.raises(LeakageError):
        _replay(synthetic_trial, clean_repo, _settings())
    assert not synthetic_trial["out_dir"].exists()


@needs_comfi
def test_own_fold_map_runs_and_is_recorded(synthetic_trial, clean_repo):
    _stamp(synthetic_trial["views"], fold=0)          # 1012 is in fold 0

    _replay(synthetic_trial, clean_repo, _settings())

    info = json.loads((synthetic_trial["out_dir"] / "run_info.json").read_text())
    assert info["landmark_map"]["fold"] == 0
    assert info["landmark_map"]["front_end"] == "nlf"
    assert "1012" not in info["landmark_map"]["fitted_on"]
    assert len(info["landmark_map"]["fitted_on"]) == 15
    assert info["landmark_map"]["in_sample"] is False
    assert len(info["landmark_map"]["sha256"]) == 64


@needs_comfi
def test_submitted_map_runs_only_when_asked_for(synthetic_trial, clean_repo):
    from rtcosmik.landmark_maps import LeakageError
    with pytest.raises(LeakageError):
        _replay(synthetic_trial, clean_repo, _settings("out_of_fold"))

    _replay(synthetic_trial, clean_repo, _settings("submitted"))

    info = json.loads((synthetic_trial["out_dir"] / "run_info.json").read_text())
    assert info["landmark_map"]["fold"] is None
    assert info["landmark_map"]["in_sample"] is True
