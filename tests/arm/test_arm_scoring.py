import copy
import csv

import numpy as np
import pytest

from .conftest import REFERENCE_RUN, build_trial, needs_comfi

LAG = 8


def _settings():
    from rtcosmik.config_loader import settings
    s = copy.deepcopy(settings)
    s.ik_type = "sbs"
    s.paper["landmark_map"] = "submitted"      # synthetic trials predate fold maps
    s.order = None                             # no low-pass: any lag is the injected one
    s.paper["clock_offset_frames"] = 0         # built from the reference: no clock offset
    return s


def _replay(trial, repo):
    from replay_nlf_arms import replay_trial
    arm = {"name": "nlf", "tag": "nlf_0-2-4-6", "reconstruction": "fuse3d",
           "cameras": trial["cameras"]}
    return replay_trial(trial["dataset"], trial["views"], trial["out_dir"], trial["participant"],
                        trial["task"], arm, _settings(), repo=repo)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    import subprocess
    repo = tmp_path_factory.mktemp("repo")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    out = {}
    for lag in (0, LAG):
        trial = build_trial(tmp_path_factory.mktemp(f"lag{lag}"), frames=400, lag=lag,
                            source=REFERENCE_RUN)
        out[lag] = (_replay(trial, repo), trial["out_dir"])
    return out


@needs_comfi
def test_injected_lag_counts_in_causal_error_and_alignment_removes_it(rows):
    (on_time, _), (late, _) = rows[0], rows[LAG]
    assert (on_time["marker_lag"], late["marker_lag"]) == (0, LAG)
    assert late["lag"] - on_time["lag"] == LAG
    assert late["causal_rmse_mean"] > on_time["causal_rmse_mean"] + 0.5
    assert abs(late["aligned_rmse_mean"] - on_time["aligned_rmse_mean"]) < 0.3


@needs_comfi
def test_bias_and_sd_decompose_the_rmse(rows):
    _, run_dir = rows[LAG]
    dofs = list(csv.DictReader(open(run_dir / "per_dof.csv")))
    assert len(dofs) == 29
    for kind in ("causal", "aligned"):
        rmse, bias, sd = (np.array([float(d[f"{kind}_{k}_deg"]) for d in dofs])
                          for k in ("rmse", "bias", "sd"))
        np.testing.assert_allclose(rmse ** 2, bias ** 2 + sd ** 2, rtol=1e-9)


@needs_comfi
def test_row_reports_markers_jitter_failures_and_limits(rows):
    row, _ = rows[0]
    for key in ("causal_marker_mm", "aligned_marker_mm", "jitter_deg", "solver_failures",
                "violation_pct", "dropped_frames"):
        assert row[key] != "" and np.isfinite(float(row[key])), key
    assert row["solver_failures"] == 0
