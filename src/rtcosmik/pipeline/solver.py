"""The human model, its calibration, and the per-frame IK, in one place.

Every entry point that turns markers into joint angles needs the same sequence:
build the model for the subject, scale and register it against the first frame,
solve once to calibrate, then solve every frame after that. That sequence used
to be copy-pasted into the offline script, the online pipeline process and the
ROS bridge, so a change to the IK API broke whichever copies were not updated
together. :class:`HumanSolver` owns it once; callers keep only what is genuinely
theirs -- reading frames, drawing, publishing, saving.
"""

import logging
from collections import deque

import numpy as np
import pinocchio as pin
import example_robot_data as robex

from rtcosmik.human_model.model_utils import (
    scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space)
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS

LOGGER = logging.getLogger(__name__)


class HumanSolver:
    """Calibrate a human model to a subject, then solve its pose each frame.

    Typical use::

        solver = HumanSolver(settings)
        for markers in stream:                 # markers: name -> (3,) position
            q = solver.solve(markers)          # calibrates on the first call

    ``solve`` calibrates on the first frame and tracks on every one after, so a
    caller does not have to carry a ``first_sample`` flag. The model, its data
    and the visual/collision models are exposed for callers that draw or export
    them; they only exist once the first frame has been seen.

    Args:
        settings: the RT-COSMIK settings object.
        gender: subject gender, defaulting to ``settings.human_gender``.
        height: subject height in metres, defaulting to ``settings.human_height``.
        weight: subject weight in kg, defaulting to ``settings.human_weight``
            when that exists.
        logger: optional logger.
    """

    def __init__(self, settings, gender=None, height=None, weight=None, logger=None):
        self.settings = settings
        self.gender = gender if gender is not None else settings.human_gender
        self.height = height if height is not None else settings.human_height
        if weight is not None:
            self.weight = weight
        else:
            self.weight = getattr(settings, "human_weight", None)
        self.logger = logger or LOGGER

        self.model = None
        self.data = None
        self.visual_model = None
        self.collision_model = None
        self.calibrated = False

        self._ik = None
        self._x = None
        self._u = None
        self._window = None

    # -- properties -------------------------------------------------------

    @property
    def joint_names(self):
        """Model joint names, or an empty list before calibration."""
        return list(self.model.names) if self.model is not None else []

    # -- public API -------------------------------------------------------

    def solve(self, mks_dict):
        """Return the configuration ``q`` that best matches this frame's markers.

        Calibrates the model on the first call, tracks on every later one.
        """
        if not self.calibrated:
            return self.calibrate(mks_dict)
        return self.step(mks_dict)

    def calibrate(self, mks_dict):
        """Build and calibrate the model against one frame, returning its ``q``."""
        human = robex.human.HumanLoader(
            height=self.height, weight=self.weight, gender=self.gender).robot
        self.model = human.model
        self.collision_model = human.collision_model
        self.visual_model = human.visual_model

        self.model = scale_human_model(
            self.model, mks_dict, gender=self.gender, subject_height=self.height)
        self.model = mks_registration(
            self.model, mks_dict, gender=self.gender, subject_height=self.height)

        ik_type = self.settings.ik_type
        if ik_type == "sbs":
            q = self._calibrate_sbs(mks_dict)
        elif ik_type == "mhe":
            q = self._calibrate_mhe(mks_dict)
        else:
            raise ValueError(
                "Invalid ik type, should be sbs (sample by sample) or mhe "
                "(moving horizon estimation)")

        self.data = self.model.createData()
        self.calibrated = True
        self.logger.info("[INFO] Model calibration finished, ready to process...")
        return q

    def step(self, mks_dict):
        """Solve one frame against the calibrated model."""
        if not self.calibrated:
            raise RuntimeError("HumanSolver.step called before calibrate")
        if self.settings.ik_type == "sbs":
            self._ik._dict_m = mks_dict
            q = self._ik.solve_ik_sample_quadprog()
            self._ik._q0 = q
            return q
        return self._solve_mhe(mks_dict)

    # -- calibration internals -------------------------------------------

    def _calibrate_sbs(self, mks_dict):
        omega = {key: 1 for key in self.settings.keys_to_track_list}
        q = pin.neutral(self.model)
        self._ik = RT_IK(self.model, mks_dict, q, self.settings.keys_to_track_list,
                         self.settings.dt, omega)
        q = self._ik.solve_ik_sample_casadi()
        self._ik._q0 = q

        # Recalibrate briefly the markers translation in joint frames
        self.model = recalibrate_marker_frames_in_joint_space(
            self.model, q, mks_dict, self.settings.marker_names)
        self._ik = RT_IK(self.model, mks_dict, q, self.settings.keys_to_track_list,
                         self.settings.dt, omega)
        return q

    def _calibrate_mhe(self, mks_dict):
        settings = self.settings
        self._ik = RT_SWIKA_FATROP(self.model, settings.keys_to_track_list,
                                   settings.N, code=settings.ik_code)
        self._x = np.zeros((self.model.nq + self.model.nv, settings.N))
        self._x[6, :] = 1
        self._u = np.zeros((self.model.nv, settings.N))
        self._window = deque(maxlen=settings.N)
        for _ in range(settings.N):
            self._window.append(mks_dict)

        q = self._solve_mhe(mks_dict, append=False)

        # Recalibrate briefly the markers translation in joint frames
        self.model = recalibrate_marker_frames_in_joint_space(
            self.model, q, mks_dict, settings.marker_names)
        self._ik = self._build_mhe_solver()
        return q

    def _build_mhe_solver(self):
        """The steady-state MHE solver, on whichever backend is configured.

        Both backends solve the same problem and share a ``solve`` signature, so
        they are interchangeable here.
        """
        settings = self.settings
        if settings.mhe_backend == "acados":
            return RT_SWIKA_ACADOS(
                self.model, settings.keys_to_track_list, settings.N, settings.dt,
                export_dir=settings.acados_export_dir,
                acados_source_dir=settings.acados_source_dir,
                max_iter=settings.mhe_max_iter)
        return RT_SWIKA_FATROP(
            self.model, settings.keys_to_track_list, settings.N,
            code=settings.ik_code, max_iter=settings.mhe_max_iter)

    def _solve_mhe(self, mks_dict, append=True):
        settings = self.settings
        if append:
            self._window.append(mks_dict)
        measurements = np.array([
            np.hstack([frame[marker] for marker in settings.keys_to_track_list])
            for frame in self._window]).T

        self._x, self._u = self._ik.solve(
            self._x, self._u, measurements, self._x[:, -1],
            settings.cost_weights, settings.dt)

        q = pin.neutral(self.model)
        q[:] = np.array(self._x[:self.model.nq, -1]).flatten()
        return q
