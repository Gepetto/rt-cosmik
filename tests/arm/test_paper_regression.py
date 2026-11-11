"""The submitted results stay reproducible. Opt in with ``RTCOSMIK_REGRESSION=1``."""
import os

import numpy as np
import pytest

from .conftest import COMFI, PARTICIPANT, TASK, CAMERAS, TRUTH_RUN

pytestmark = pytest.mark.skipif(os.environ.get("RTCOSMIK_REGRESSION") != "1",
                                reason="set RTCOSMIK_REGRESSION=1 to run")


def _angles(run_dir):
    table = np.genfromtxt(run_dir / "joint_angles.csv", delimiter=",", names=True)
    names = [n for n in table.dtype.names if n.endswith("rad")]
    return np.stack([table[n] for n in names], 1)


def _rms_deg(a, b):
    n = min(len(a), len(b))
    return np.degrees(np.sqrt(np.mean((a[:n] - b[:n]) ** 2, axis=0)))


@pytest.fixture(scope="module")
def settings():
    """The paper settings with the submitted map, which these runs reproduce."""
    import copy
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.paper["landmark_map"] = "submitted"
    s.paper["clock_offset"] = 0
    return s


@pytest.fixture(scope="module")
def direct_run(settings, tmp_path_factory):
    """NLF, four cameras, the paper settings, recording its views."""
    import sweep
    root = tmp_path_factory.mktemp("direct")
    out_dir, views = root / "run", root / "views"
    sweep.run_nlf(str(COMFI), PARTICIPANT, TASK, CAMERAS, out_dir, settings, views_cache=views)
    return out_dir, views / PARTICIPANT / f"{TASK}.npz"


def test_direct_run_reproduces_the_submitted_run(direct_run):
    out_dir, _ = direct_run
    diff = _rms_deg(_angles(out_dir), _angles(TRUTH_RUN))
    assert len(_angles(out_dir)) == len(_angles(TRUTH_RUN))
    assert diff.mean() < 0.1, f"mean per-DoF RMS difference {diff.mean():.3f} deg"


def test_replay_reproduces_the_direct_run(direct_run, settings, tmp_path):
    from replay_nlf_arms import replay_trial
    out_dir, views = direct_run
    arm = {"name": "nlf", "tag": "nlf_0-2-4-6", "reconstruction": "fuse3d",
           "cameras": CAMERAS}
    replay_trial(COMFI, views, tmp_path / "replay", PARTICIPANT, TASK, arm, settings)
    diff = _rms_deg(_angles(tmp_path / "replay"), _angles(out_dir))
    assert diff.max() < 1e-3, f"max per-DoF RMS difference {diff.max():.5f} deg"


def test_direct_run_scores_like_the_submitted_run(direct_run):
    from rtcosmik.paper.scoring import reference_dir, score_trial
    out_dir, _ = direct_run
    reference = reference_dir(COMFI, PARTICIPANT, TASK)
    ours, submitted = (score_trial(run, reference, clock_offset=0)[0]
                       for run in (out_dir, TRUTH_RUN))
    assert ours["scored_dof"] == submitted["scored_dof"] == 29
    assert abs(ours["aligned_rmse_mean"] - submitted["aligned_rmse_mean"]) < 0.05
