import copy
import json
import subprocess

import pytest

from .conftest import needs_comfi


def _per_frame_settings():
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.ik_type = "sbs"
    return s


def _replay(trial, repo, arm=None):
    from replay_nlf_arms import replay_trial
    arm = arm or {"name": "nlf", "tag": "nlf_0-2-4-6", "reconstruction": "fuse3d",
                  "cameras": trial["cameras"], "fold": None}
    return replay_trial(trial["dataset"], trial["views"], trial["out_dir"], trial["participant"],
                        trial["task"], arm, _per_frame_settings(), repo=repo)


@needs_comfi
def test_run_info_records_provenance(synthetic_trial, clean_repo):
    _replay(synthetic_trial, clean_repo)

    info = json.loads((synthetic_trial["out_dir"] / "run_info.json").read_text())
    head = subprocess.run(["git", "-C", str(clean_repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert info["code"]["commit"] == head
    assert info["arm"]["tag"] == "nlf_0-2-4-6"
    assert info["arm"]["fold"] is None
    assert info["settings"]["ik_type"] == "sbs"
    assert len(info["landmark_map"]["sha256"]) == 64
    assert info["landmark_map"]["count"] == len(info["settings"]["nlf_indices"])
    assert info["hardware"]["cpu"]
    assert "gpu" in info["hardware"]
    assert info["started_utc"].endswith("Z")


@needs_comfi
@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_uncommitted_pipeline_code_is_refused(synthetic_trial, clean_repo, change):
    from rtcosmik.provenance import DirtyTreeError
    if change == "modified":
        (clean_repo / "settings.py").write_text("N = 10\n")
    else:
        (clean_repo / "src" / "new_arm.py").write_text("y = 2\n")

    with pytest.raises(DirtyTreeError):
        _replay(synthetic_trial, clean_repo)
    assert not synthetic_trial["out_dir"].exists()


@needs_comfi
def test_uncommitted_docs_do_not_block_a_run(synthetic_trial, clean_repo):
    (clean_repo / "NOTES.md").write_text("draft\n")

    row = _replay(synthetic_trial, clean_repo)

    assert row["status"] == "ok"
    assert (synthetic_trial["out_dir"] / "joint_angles.csv").exists()
