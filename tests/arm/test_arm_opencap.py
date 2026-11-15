"""Calibrate on the OpenCap static trial, then start a trial that begins seated."""
import copy
from pathlib import Path

import numpy as np
import pytest


SUBJECT = Path("/root/workspace/OpenCap/data/LabValidation_withVideos/subject4")
needs_opencap = pytest.mark.skipif(not SUBJECT.is_dir(), reason="OpenCap LabValidation not mounted")


@pytest.fixture(scope="module")
def solver():
    from rtcosmik.config_loader import settings
    from rtcosmik.paper import opencap as oc
    from rtcosmik.pipeline.solver import HumanSolver
    s = oc.dataset_settings(copy.deepcopy(settings), oc.REFERENCE_SET)
    s.mhe_backend, s.ik_code = "fatrop", "python"      # no generated artefact needed
    static = oc.reference_markers(oc.trc_path(SUBJECT, oc.STATIC))
    solver = HumanSolver(s, gender="f", height=1.68, weight=62.6)
    q = solver.calibrate({name: np.nanmedian(xyz, axis=0) for name, xyz in static.items()})
    return solver, s, q


@needs_opencap
def test_static_only_markers_scale_the_model_without_being_tracked(solver):
    from rtcosmik.paper import opencap as oc
    human, settings, q = solver
    frames = {f.name for f in human.model.frames}
    assert set(settings.marker_names) <= frames
    assert not set(oc.STATIC_ONLY) & frames
    oc.require_standing(settings.joint_angles_names, q)
    shank = np.linalg.norm(human.model.jointPlacements[human.model.getJointId("right_ankle_Z")].translation)
    assert 0.33 < shank < 0.45                          # scaled from the medial markers, 1.68 m tall


@needs_opencap
def test_trial_starting_seated_is_fitted_on_its_first_frame(solver):
    import pinocchio as pin
    from rtcosmik.paper import opencap as oc
    human, settings, _ = solver
    seated = oc.reference_markers(oc.trc_path(SUBJECT, "STS1"), at=[0.5])
    frame = {name: seated[name][0] for name in settings.marker_names}

    q = human.start_trial(frame)

    data = human.model.createData()
    pin.framesForwardKinematics(human.model, data, q)
    gaps = [np.linalg.norm(data.oMf[human.model.getFrameId(n)].translation - frame[n])
            for n in settings.marker_names]
    assert np.sqrt(np.mean(np.square(gaps))) < 0.02
    names = settings.joint_angles_names
    knee = np.degrees(q[names.index("Right_Knee_Flexion_Extension[rad]")])
    assert abs(knee) > 60                               # seated, far from the standing calibration
    np.testing.assert_allclose(human.step(frame), q, atol=0.02)      # no transient on the next frame
