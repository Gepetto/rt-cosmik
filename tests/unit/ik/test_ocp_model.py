"""Unit tests for the subject-independent OCP geometry.

These run in seconds and need no solver, no compiler, no dataset and no marker
capture: the structural model is topology only. They cover the two properties the
whole pre-generation scheme rests on:

  * the parameter vector is ordered identically at build and at runtime, and
  * two different people produce the *same* fingerprint, which is what makes one
    generated artefact valid for both.

The end-to-end behaviour (a generated solver actually reproducing a per-subject
one) is covered separately by tests/benchmark/validate_ocp_params.py, which needs
a compiler and a dataset.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.config_loader import settings          # noqa: E402
from rtcosmik.ik import ocp_model                    # noqa: E402


@pytest.fixture(scope="module")
def structural_model():
    return ocp_model.build_structural_model(settings)


@pytest.fixture(scope="module")
def keys(structural_model):
    return [k for k in settings.keys_to_track_list if structural_model.existFrame(k)]


@pytest.fixture(scope="module")
def parameterized(structural_model, keys):
    return ocp_model.parameterize(structural_model, keys)


def test_parameterizes_translations_only(structural_model, keys, parameterized):
    """Three scalars per internal joint and per tracked marker, freeflyer aside."""
    _, p_sym, p_default, joint_ids, frame_ids = parameterized
    internal = [j for j in range(1, structural_model.njoints)
                if structural_model.joints[j].nq != 7]
    assert joint_ids == internal
    assert len(frame_ids) == len(keys)
    assert p_sym.shape[0] == 3 * (len(joint_ids) + len(frame_ids))
    assert p_default.shape == (p_sym.shape[0],)


def test_forward_kinematics_depends_on_the_parameters(parameterized):
    import casadi
    cmodel, p_sym, _, _, frame_ids = parameterized
    q = casadi.SX.sym("q", cmodel.nq)
    fk = ocp_model.marker_fk_expr(cmodel, q, frame_ids)
    assert casadi.depends_on(fk, p_sym)


def test_dynamics_does_not_depend_on_the_parameters(parameterized):
    """Only the cost needs geometry; integration is on the configuration manifold.

    If this ever fails, the discrete dynamics has gained a parameter dependence
    and the generated code is no longer subject-independent.
    """
    import casadi
    import pinocchio.casadi as cpin
    cmodel, p_sym, _, _, _ = parameterized
    q = casadi.SX.sym("q", cmodel.nq)
    v = casadi.SX.sym("v", cmodel.nv)
    assert not casadi.depends_on(cpin.integrate(cmodel, q, v), p_sym)


def test_extract_params_reproduces_the_defaults(structural_model, parameterized):
    """Build-time and runtime orderings must agree scalar for scalar."""
    _, _, p_default, joint_ids, frame_ids = parameterized
    extracted = ocp_model.extract_params(structural_model, joint_ids, frame_ids)
    assert np.array_equal(extracted, p_default)


def test_missing_tracked_marker_is_refused(structural_model, keys):
    with pytest.raises(ValueError, match="not a frame in the model"):
        ocp_model.parameterize(structural_model, list(keys) + ["NOT_A_MARKER"])


# --- the property that makes artefact reuse valid ---------------------------

def _describe(model, keys, dt=None):
    _, _, _, joint_ids, frame_ids = ocp_model.parameterize(model, keys)
    return ocp_model.describe(model, keys, settings.N, dt, True,
                              joint_ids, frame_ids)


def test_two_different_people_share_a_fingerprint(keys):
    """A short female and a tall male must hash identically.

    Their segment lengths differ, but the topology, rotations, limits and frame
    set do not -- which is exactly why one generated OCP serves both.
    """
    short = ocp_model.build_structural_model(settings, gender="f", height=1.55,
                                             weight=45.0)
    tall = ocp_model.build_structural_model(settings, gender="m", height=1.95,
                                            weight=95.0)
    assert ocp_model.fingerprint(_describe(short, keys)) == \
        ocp_model.fingerprint(_describe(tall, keys))

    # ...and they really are different people.
    _, _, _, joint_ids, frame_ids = ocp_model.parameterize(short, keys)
    a = ocp_model.extract_params(short, joint_ids, frame_ids)
    b = ocp_model.extract_params(tall, joint_ids, frame_ids)
    assert np.abs(a - b).max() > 1e-3


@pytest.mark.parametrize("mutate,label", [
    (lambda d: d.update(N=d["N"] + 1), "N"),
    (lambda d: d.update(keys_to_track=d["keys_to_track"][:-1]), "tracked markers"),
    (lambda d: d.update(with_freeflyer=False), "freeflyer"),
    (lambda d: d.update(lower_limit=[v - 1 for v in d["lower_limit"]]), "limits"),
])
def test_fingerprint_catches_a_changed_formulation(structural_model, keys, mutate, label):
    base = _describe(structural_model, keys)
    changed = json.loads(json.dumps(base))
    mutate(changed)
    assert ocp_model.fingerprint(base) != ocp_model.fingerprint(changed), label


def test_dt_is_recorded_for_acados_and_not_for_fatrop(structural_model, keys):
    """acados bakes dt into the discrete dynamics; fatrop takes it at runtime.

    Recording dt for fatrop would discard a valid artefact on every framerate
    change, so the two backends deliberately describe themselves differently.
    """
    acados = _describe(structural_model, keys, dt=settings.dt)
    fatrop = _describe(structural_model, keys, dt=None)
    assert acados["dt"] == pytest.approx(settings.dt)
    assert fatrop["dt"] is None

    faster = _describe(structural_model, keys, dt=settings.dt * 2)
    assert ocp_model.fingerprint(acados) != ocp_model.fingerprint(faster)
    assert ocp_model.fingerprint(fatrop) == \
        ocp_model.fingerprint(_describe(structural_model, keys, dt=None))


# --- manifest ----------------------------------------------------------------

def test_manifest_round_trip_and_drift_detection(structural_model, keys):
    base = _describe(structural_model, keys, dt=settings.dt)
    with tempfile.TemporaryDirectory() as directory:
        ocp_model.write_manifest(directory, base, "acados")
        assert ocp_model.check_manifest(directory, base, "acados") is not None

        drifted = json.loads(json.dumps(base))
        drifted["dt"] = base["dt"] * 2
        with pytest.raises(RuntimeError) as excinfo:
            ocp_model.check_manifest(directory, drifted, "acados")
        message = str(excinfo.value)
        # The message has to name the offending field: a silently mismatched
        # artefact solves happily on the wrong skeleton.
        assert "dt" in message
        assert "run_ocp_codegen" in message


def test_missing_manifest_is_an_error_not_a_silent_reuse(structural_model, keys):
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(RuntimeError, match="cannot be verified"):
            ocp_model.check_manifest(directory, _describe(structural_model, keys), "fatrop")


def test_artifact_root_is_absolute_and_overridable(monkeypatch):
    """It must not depend on the working directory: the previous acados default
    was os.getcwd()/acados_codegen, so launching from elsewhere silently
    recompiled."""
    monkeypatch.delenv("RTCOSMIK_OCP_DIR", raising=False)
    assert os.path.isabs(ocp_model.artifact_root(settings))
    monkeypatch.setenv("RTCOSMIK_OCP_DIR", "/tmp/some_ocp_dir")
    assert ocp_model.artifact_root(settings) == "/tmp/some_ocp_dir"
    # The profile is part of the path: each is a separate generated artefact.
    assert ocp_model.backend_dir("acados", settings, "realtime") == \
        "/tmp/some_ocp_dir/acados/realtime"
    assert ocp_model.backend_dir("acados", settings, "accurate") == \
        "/tmp/some_ocp_dir/acados/accurate"


def test_profiles_are_defined_for_both_backends():
    for backend in ("acados", "fatrop"):
        for profile in ("realtime", "accurate"):
            assert ocp_model.profile_options(backend, profile)
    with pytest.raises(ValueError, match="unknown solver profile"):
        ocp_model.profile_options("acados", "nonsense")


def test_profile_changes_the_fingerprint(structural_model, keys):
    """Otherwise a profile switch would silently reuse the wrong .so.

    nlp_solver_type, qp_solver and globalization are baked into the generated C,
    so the profile is not a runtime option.
    """
    _, _, _, joint_ids, frame_ids = ocp_model.parameterize(structural_model, keys)
    digests = set()
    for profile in ("realtime", "accurate"):
        description = ocp_model.describe(
            structural_model, keys, settings.N, settings.dt, True,
            joint_ids, frame_ids,
            solver_options=ocp_model.profile_options("acados", profile))
        digests.add(ocp_model.fingerprint(description))
    assert len(digests) == 2
