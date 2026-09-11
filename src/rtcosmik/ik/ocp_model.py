"""Subject-independent OCP geometry, shared by the fatrop and acados MHE backends.

Both backends bake the human's forward kinematics into generated code, so both
recompile whenever a new person is calibrated -- 20-40 s, in the middle of a live
session. Everything that differs between people is a *translation*: segment
lengths (``jointPlacements[j].translation``) and marker offsets
(``frames[f].placement.translation``). Turning those into solver parameters makes
one generated artefact serve everybody.

Measured on the real 43-dof model, 29 tracked markers (195 parameters):

    FK regression, parameterized-at-default vs baked ......... 0.0
    subject A's generated FK fed subject B's p, vs B's baked .. 0.0
    acados solve time, baked vs parameterized ................ 0.59 -> 0.60 ms

See ``docs/acados_mhe_ik.md`` for the full measurements.

The dangerous failure mode is not a crash but *silent misalignment*: change
``keys_to_track_list``, ``SGTS_MKS_MAPPING`` or the URDF and a stale artefact
still loads, still solves, and quietly computes another skeleton. Nothing raises.
:func:`fingerprint` exists so that becomes a checked error instead.
"""

import hashlib
import json
import logging
import os

import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

LOGGER = logging.getLogger(__name__)

#: Manifest filename written beside every generated OCP artefact.
MANIFEST_NAME = "ocp_manifest.json"


#: Speed/accuracy configurations, one generated artefact each.
#:
#: Measured on 120 frames of real marker data, 43 dof, N=10 (median / p95 / max
#: solve ms, marker RMSE mm, median frame-to-frame |dq|):
#:
#:   acados SQP it=50 tol=1e-4   3.66 /  89.5 / 131.7   1.92   0.0053   <- old default
#:   acados SQP it=50 tol=1e-6  11.73 / 185.0 / 212.7   1.03   0.0056
#:   acados SQP it=10 tol=1e-6  12.31 /  43.2 /  45.8   1.03   0.0056
#:   acados SQP_RTI              4.85 /   5.9 /   8.7   1.02   0.0049
#:   fatrop it=100 tol=1e-6     35.70 /  41.0 /  45.7   1.07   0.0041
#:   fatrop it=10  tol=1e-4     26.86 /  28.8 /  46.5   1.09   0.0042
#:
#: Two things this measurement settled. The old acados default was the worst of
#: both worlds: a 131 ms tail *and* the worst marker fit, because tol=1e-4 stops
#: before the marker term is properly minimised. And SQP_RTI is not the usual
#: real-time compromise here -- it matches fully converged SQP on marker RMSE
#: (1.02 vs 1.03) with slightly *less* jitter, at a twentieth of the tail.
SOLVER_PROFILES = {
    "acados": {
        # One real-time iteration per frame: bounded cost, no iteration tail.
        # nlp_solver_max_iter is inherited but inert here -- SQP_RTI takes
        # exactly one iteration (measured 4.88 ms at 50 vs 4.78 ms at 1).
        "realtime": {"nlp_solver_type": "SQP_RTI"},
        # Converged. it=10 reaches the same answer as it=50 (dq 7.6e-05) for a
        # quarter of the tail, so the extra budget only buys worst cases.
        "accurate": {"nlp_solver_type": "SQP", "nlp_solver_max_iter": 10,
                     "tol": 1e-6},
    },
    "fatrop": {
        # fatrop converges in under 10 iterations and degrades sharply below 3
        # (it=1 gives 18 mm RMSE), so its "fast" profile still allows 10.
        "realtime": {"max_iter": 10, "tol": 1e-4},
        "accurate": {"max_iter": 100, "tol": 1e-6},
    },
}


def profile_options(backend, profile):
    """Solver settings for one backend/profile pair."""
    try:
        return dict(SOLVER_PROFILES[backend][profile])
    except KeyError:
        raise ValueError(
            f"unknown solver profile {profile!r} for backend {backend!r}; "
            f"expected one of {sorted(SOLVER_PROFILES.get(backend, {}))}") from None


def parameterize(pin_model, keys_to_track):
    """Replace every subject-dependent translation with a CasADi parameter.

    Only translations are parameterized. Rotations, joint types, the kinematic
    topology and the position limits are identical for every subject -- verified
    by diffing a 1.64 m/51 kg model against a 1.87 m/90 kg one, where all of
    those differ by exactly 0.

    The freeflyer is skipped: its translation is a constant zero and its rotation
    carries the enforced freeflyer orientation. Only the *tracked* marker frames
    are parameterized, since the other registered frames never enter the cost.

    Args:
        pin_model: a structural ``pin.Model`` with the marker frames registered.
            Its numbers become the defaults; only its structure matters.
        keys_to_track: tracked marker names, in the order the cost stacks them.

    Returns:
        tuple: ``(cmodel, p_sym, p_default, joint_ids, frame_ids)``. ``p_sym`` is
        the stacked symbol, ``p_default`` the numeric values from ``pin_model``,
        and the two id lists fix the ordering that :func:`extract_params` must
        reproduce exactly.
    """
    cmodel = cpin.Model(pin_model)
    symbols, defaults, joint_ids, frame_ids = [], [], [], []

    for jid in range(1, pin_model.njoints):
        if pin_model.joints[jid].nq == 7:      # freeflyer
            continue
        length = casadi.SX.sym(f"L_{jid}", 3)
        # The rotation must be read from the NUMERIC model: the cpin.Model's own
        # rotation is already an SX, and np.array() on it raises.
        rotation = casadi.SX(np.array(pin_model.jointPlacements[jid].rotation))
        cmodel.jointPlacements[jid] = cpin.SE3(rotation, length)
        symbols.append(length)
        defaults.append(np.asarray(pin_model.jointPlacements[jid].translation, float))
        joint_ids.append(jid)

    for key in keys_to_track:
        if not pin_model.existFrame(key):
            raise ValueError(
                f"tracked marker {key!r} is not a frame in the model. The generated "
                "OCP has a fixed residual size, so the tracked set cannot vary; "
                "register every marker before generating.")
        fid = pin_model.getFrameId(key)
        offset = casadi.SX.sym(f"off_{fid}", 3)
        frame = cmodel.frames[fid]
        frame.placement = cpin.SE3(
            casadi.SX(np.array(pin_model.frames[fid].placement.rotation)), offset)
        cmodel.frames[fid] = frame           # get-modify-set; in-place does not stick
        symbols.append(offset)
        defaults.append(np.asarray(pin_model.frames[fid].placement.translation, float))
        frame_ids.append(int(fid))

    return (cmodel, casadi.vertcat(*symbols), np.concatenate(defaults),
            joint_ids, frame_ids)


def extract_params(pin_model, joint_ids, frame_ids):
    """Numeric parameter vector for a calibrated model.

    The ordering must match :func:`parameterize` scalar for scalar; the id lists
    it returned are what guarantees that, so pass them through rather than
    recomputing them.
    """
    values = [np.asarray(pin_model.jointPlacements[j].translation, float)
              for j in joint_ids]
    values += [np.asarray(pin_model.frames[f].placement.translation, float)
               for f in frame_ids]
    return np.concatenate(values)


def marker_fk_expr(cmodel, cq, frame_ids):
    """Stacked ``[x0,y0,z0, x1,...]`` marker positions for configuration ``cq``."""
    cdata = cmodel.createData()
    cpin.framesForwardKinematics(cmodel, cdata, cq)
    return casadi.vertcat(*[cdata.oMf[fid].translation for fid in frame_ids])


def describe(pin_model, keys_to_track, N, dt, with_freeflyer, joint_ids, frame_ids,
             solver_options=None):
    """Everything baked into generated code, as a comparable dict.

    Args:
        dt: the timestep, or **None** when the backend takes it as a runtime
            input. acados bakes dt into ``disc_dyn_expr`` via
            ``cpin.integrate(cm, cq, cdq*dt)``, so a dt change invalidates its
            artefact. fatrop declares dt as an ``opti.parameter()`` passed
            through ``to_function``, so its artefact is dt-independent --
            verified: the same compiled solver returns different trajectories for
            dt=0.025 and dt=0.100. Putting dt in fatrop's fingerprint would throw
            away a valid 200 s build every time the framerate changed.
    """
    return {
        "joint_names": [str(n) for n in pin_model.names],
        "joint_types": [pin_model.joints[j].shortname()
                        for j in range(pin_model.njoints)],
        "frame_names": [f.name for f in pin_model.frames],
        "keys_to_track": [str(k) for k in keys_to_track],
        "param_joint_ids": [int(j) for j in joint_ids],
        "param_frame_ids": [int(f) for f in frame_ids],
        "nq": int(pin_model.nq),
        "nv": int(pin_model.nv),
        "N": int(N),
        "dt": None if dt is None else round(float(dt), 12),
        "with_freeflyer": bool(with_freeflyer),
        # Baked into the generated C: nlp_solver_type, qp_solver and
        # globalization all change what is emitted, so a speed/accuracy profile
        # change must invalidate the artefact rather than silently reuse it.
        "solver_options": (None if solver_options is None
                           else {k: (round(v, 12) if isinstance(v, float) else v)
                                 for k, v in sorted(solver_options.items())}),
        "lower_limit": np.asarray(pin_model.lowerPositionLimit, float).round(12).tolist(),
        "upper_limit": np.asarray(pin_model.upperPositionLimit, float).round(12).tolist(),
    }


def fingerprint(description):
    """Stable hash of a :func:`describe` dict."""
    return hashlib.sha256(
        json.dumps(description, sort_keys=True).encode()).hexdigest()


def artifact_root(settings=None):
    """Where generated OCP artefacts live.

    A fixed absolute path, deliberately: the previous default was
    ``os.getcwd()/acados_codegen``, so *where the pipeline was launched from*
    decided whether a compiled solver was found, and running from elsewhere
    silently recompiled.
    """
    override = os.getenv("RTCOSMIK_OCP_DIR", "").strip()
    if override:
        return os.path.abspath(override)
    if settings is not None and getattr(settings, "acados_export_dir", None):
        return os.path.abspath(settings.acados_export_dir)
    base = getattr(settings, "cosmik_path", None) if settings is not None else None
    if base is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base, "ocp")


def backend_dir(backend, settings=None, profile=None):
    """Artefact directory for one backend and speed/accuracy profile.

    The profile is part of the path because ``nlp_solver_type``, ``qp_solver``
    and ``globalization`` are baked into the generated C -- switching profiles is
    a different artefact, not a runtime option.
    """
    if profile is None and settings is not None:
        profile = getattr(settings, "mhe_profile", "realtime")
    return os.path.join(artifact_root(settings), backend, profile or "realtime")


def write_manifest(directory, description, backend, extra=None):
    """Record what a generated artefact was built from."""
    os.makedirs(directory, exist_ok=True)
    payload = {"backend": backend,
               "fingerprint": fingerprint(description),
               "description": description}
    if extra:
        payload.update(extra)
    path = os.path.join(directory, MANIFEST_NAME)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_manifest(directory):
    """The manifest beside an artefact, or None when there is none."""
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _diff(expected, found):
    """Human-readable list of what changed between two descriptions."""
    changes = []
    for key in sorted(set(expected) | set(found)):
        a, b = expected.get(key), found.get(key)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
            changes.append(f"{key}: {len(b)} entries generated, {len(a)} now")
        else:
            sa, sb = str(a), str(b)
            if len(sa) > 60:
                sa, sb = sa[:57] + "...", sb[:57] + "..."
            changes.append(f"{key}: generated {sb}, now {sa}")
    return changes


def check_manifest(directory, description, backend):
    """Refuse to reuse an artefact that was generated from something else.

    Raises:
        RuntimeError: no manifest, or the configuration has drifted. The message
            names what changed, because the alternative is a solver that runs
            happily on the wrong skeleton.
    """
    manifest = read_manifest(directory)
    if manifest is None:
        raise RuntimeError(
            f"No {MANIFEST_NAME} in {directory}: the generated OCP cannot be "
            f"verified against the current configuration.\n"
            f"  Regenerate with: python3 scripts/python/core/run_ocp_codegen.py "
            f"--backend {backend}")
    if manifest.get("fingerprint") == fingerprint(description):
        return manifest
    changes = _diff(description, manifest.get("description", {}))
    detail = "\n".join(f"    - {c}" for c in changes) or "    - (unknown)"
    raise RuntimeError(
        f"The generated {backend} OCP in {directory} does not match the current "
        f"configuration:\n{detail}\n"
        f"  Regenerate with: python3 scripts/python/core/run_ocp_codegen.py "
        f"--backend {backend}")


def register_marker_frames(model):
    """Attach a frame per tracked marker, with a zero offset.

    ``mks_registration`` needs real marker positions, but only to compute each
    marker's offset in its parent joint frame -- and that offset is precisely
    what gets parameterized away. So generating the OCP needs no capture at all:
    the frames are added here with a zero offset, in the same order
    ``mks_registration`` uses, and the offsets arrive later as parameters.

    Ordering matters and comes from ``SGTS_MKS_MAPPING``, so the frame ids match
    a model built the ordinary way.
    """
    from rtcosmik.human_model.model_utils import (
        MKS_COSMIK_2_JOINTS, SGTS_MKS_MAPPING)

    inertia = pin.Inertia.Zero()
    for _, marker_names in SGTS_MKS_MAPPING.items():
        for marker_name in marker_names:
            joint_id = model.getJointId(MKS_COSMIK_2_JOINTS[marker_name])
            model.addFrame(pin.Frame(marker_name, joint_id,
                                     model.joints[joint_id].id,
                                     pin.SE3(np.eye(3), np.zeros(3)),
                                     pin.FrameType.OP_FRAME, inertia), False)
    return model


def build_structural_model(settings, gender=None, height=None, weight=None):
    """The model an OCP is generated from: correct structure, no real numbers.

    Only the topology, joint types, rotations, position limits and the frame set
    reach the generated code -- verified: a deliberately bogus marker set yields
    a byte-identical fingerprint and symbolic FK. Every translation is a
    parameter, supplied per subject by ``set_model_params``.

    So this deliberately skips ``scale_human_model`` and ``mks_registration``:
    both exist to compute translations, and both would need a marker capture to
    do it. Nothing about a real person is needed, or wanted, here.
    """
    import example_robot_data as robex

    gender = gender or settings.human_gender
    height = height if height is not None else settings.human_height
    weight = weight if weight is not None else settings.human_weight

    model = robex.human.HumanLoader(height=height, weight=weight,
                                    gender=gender).robot.model
    return register_marker_frames(model)
