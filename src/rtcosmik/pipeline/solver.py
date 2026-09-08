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
    scale_human_model, mks_registration, recalibrate_marker_frames_in_joint_space,
    apply_joint_locks)
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS
from rtcosmik.ik import ocp_model

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

        # Must happen before the OCP is built, and must match what
        # build_structural_model did when the artefact was generated -- the
        # limits are fingerprinted, so a mismatch is caught rather than ignored.
        locked = getattr(self.settings, "locked_joints", ())
        if locked:
            apply_joint_locks(self.model, locked)
            self.logger.info(
                f"[MODEL] locked {len(locked)} DoF: {', '.join(locked)}")

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
                                   settings.N)
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

        The OCP is parameterized by the subject's geometry, so a pre-generated
        artefact (``scripts/python/core/run_ocp_codegen.py``) is reused and the
        subject applied as parameters -- milliseconds instead of the 20-40 s
        regeneration this used to cost every time a new person was calibrated.
        Without a matching artefact it generates one for this subject, which is
        the old behaviour.
        """
        settings = self.settings
        directory = ocp_model.backend_dir(settings.mhe_backend, settings)
        solver, source = self._resolve_mhe_solver(directory)
        self._log_mhe_configuration(solver, source)
        return solver

    def _resolve_mhe_solver(self, directory):
        """Build the solver and say, in words, where its OCP came from."""
        settings = self.settings
        keys = settings.keys_to_track_list
        options = ocp_model.profile_options(settings.mhe_backend,
                                            settings.mhe_profile)

        if settings.mhe_backend == "acados":
            try:
                solver = RT_SWIKA_ACADOS(
                    self.model, keys, settings.N, settings.dt, build=False,
                    export_dir=directory,
                    acados_source_dir=settings.acados_source_dir,
                    solver_options=options)
                solver.set_model_params(self.model)
                return solver, f"pre-generated, reused from {directory}"
            except (RuntimeError, FileNotFoundError, OSError) as exc:
                self.logger.warning(
                    f"[WARN] Generating the acados OCP for this subject ({exc}). "
                    "Run scripts/python/core/run_ocp_codegen.py to avoid this.")
            solver = RT_SWIKA_ACADOS(
                self.model, keys, settings.N, settings.dt,
                export_dir=directory,
                acados_source_dir=settings.acados_source_dir,
                solver_options=options)
            return solver, f"COMPILED FOR THIS SUBJECT into {directory}"

        if settings.ik_code == "c":
            try:
                _, _, _, joint_ids, frame_ids = ocp_model.parameterize(self.model, keys)
                ocp_model.check_manifest(
                    directory,
                    # dt is a runtime input for fatrop, so it is not part of
                    # the artefact's identity.
                    ocp_model.describe(
                        self.model, keys, settings.N, None, True,
                        joint_ids, frame_ids,
                        solver_options={**RT_SWIKA_FATROP.DEFAULT_SOLVER_OPTIONS,
                                        **options}),
                    "fatrop")
                solver = RT_SWIKA_FATROP(
                    self.model, keys, settings.N, code="c",
                    export_dir=directory, solver_options=options)
                solver.set_model_params(self.model)
                return solver, f"pre-compiled, {solver.library_path()}"
            except (RuntimeError, FileNotFoundError, OSError) as exc:
                self.logger.warning(
                    f"[WARN] Falling back to the Python fatrop OCP ({exc}). Run "
                    "scripts/python/core/run_ocp_codegen.py --backend fatrop.")

        solver = RT_SWIKA_FATROP(
            self.model, keys, settings.N, code="python",
            export_dir=directory, solver_options=options)
        return solver, ("CasADi function built for this subject -- NOT the "
                        "compiled OCP (set ik_code='c' to use it)")

    @staticmethod
    def _describe_iterations(solver):
        """How many SQP iterations this solver will actually take.

        SQP_RTI performs exactly one by construction and ignores
        ``nlp_solver_max_iter`` -- measured: 4.88 ms at 50 versus 4.78 ms at 1,
        i.e. within noise. Reporting the inherited 50 would suggest a knob worth
        turning when there is none.
        """
        options = getattr(solver, "_solver_options", {})
        if options.get("nlp_solver_type") == "SQP_RTI":
            return "1 (SQP_RTI, fixed by the solver type)"
        return str(options.get("nlp_solver_max_iter")
                   or options.get("max_iter") or "solver default")

    def _log_mhe_configuration(self, solver, source):
        """Everything defining the IK, in one banner.

        Messages are pre-formatted rather than passed printf-style: the ROS
        bridge hands in an rclpy logger, whose ``info(message, **kwargs)`` takes
        no positional format arguments and would raise TypeError.

        Printed unconditionally on every path. The Python fatrop path used to
        return silently, so a run with ``ik_code='python'`` was indistinguishable
        from one using the compiled OCP.
        """
        settings = self.settings
        backend = settings.mhe_backend
        weights = list(settings.cost_weights)
        lines = [
            f"profile   : {settings.mhe_profile}   -> "
            f"{ocp_model.profile_options(backend, settings.mhe_profile)}",
            f"backend   : {backend}"
            + (f"   ik_code={settings.ik_code}" if backend == "fatrop" else ""),
            f"OCP       : {source}",
            f"horizon   : N={settings.N} nodes, dt={settings.dt:.4f} s "
            f"({(settings.N - 1) * settings.dt:.3f} s window)"
            + ("   [dt baked in]" if backend == "acados" else "   [dt is a runtime input]"),
            f"iterations: {self._describe_iterations(solver)}",
            f"model     : nq={self.model.nq} nv={self.model.nv}, "
            f"{len(settings.keys_to_track_list)} tracked markers, "
            f"{getattr(solver, 'n_params', '?')} geometry parameters",
            f"cost      : markers={weights[0]:g} state={weights[1]:g} "
            f"control={weights[2]:g}",
            f"subject   : height={self.height:.2f} m weight={self.weight:.1f} kg "
            f"gender={self.gender}",
        ]
        for line in lines:
            self.logger.info(f"[IK] {line}")

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
