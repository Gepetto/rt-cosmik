"""Fit a SMPL body to NLF's dense vertex output, as a denoising step before IK.

NLF regresses each canonical SMPL-X vertex independently, so nothing in its
output forces the result to be a body a human could have: segment lengths drift
frame to frame and the surface distorts locally. Fitting the SMPL model back to
those points reimposes that constraint, and the earlier arms in this study say
it should matter -- what predicts joint-angle accuracy is the *shape* of the
marker cloud, not its absolute position (Spearman 0.95 against 0.45), and the
one arm with a genuinely good body model behind it won by a clear margin.

The correspondence is free: NLF's canonical vertices *are* SMPL-X vertices, index
for index, so its dense output is a valid fitter input with no registration step.
The 35 parity markers are then the same ``settings.nlf_indices`` rows of the
fitted vertices that the plain NLF arm reads off the raw ones, which keeps the
two arms on exactly the same marker convention.

Three shape policies, because which one is right is an empirical question:

``free``
    Betas re-estimated every frame. Most adaptive, least constrained -- and it
    lets the body change size between frames, which is exactly the artefact the
    fit is supposed to remove.
``calibrated``
    Betas fitted once over the first ``calibration_frames`` frames with
    ``share_beta``, then held fixed and only the pose solved. Causal, and closest
    to how a real session would run: the subject is measured once, then tracked.
``shared``
    Betas shared across the whole trial, fitted offline over all frames. Not
    causal, so it cannot ship -- it exists as the upper bound the calibrated mode
    is trying to approach.

**Fitting a vertex subset does not work here, and was measured rather than
assumed.** smplfitter can build a body model over a subset of vertices, which
would cut both the fit and NLF's output cost, and it looks like the obvious
saving: only 35 of the 10475 vertices are ever used downstream. But the fit
collapses. Against an exact synthetic target the full model converges to 0.65 mm
while a 1024-vertex decimated subset plus the markers reaches only 148.9 mm at 4
iterations, 81.3 at 8 and 65.6 at 16 -- so it is not a convergence or
regularisation problem. smplfitter's own ``vertex_subset_size`` path is no better
(109.3 mm at 512): it wants a precomputed
``vertex_subset_joint_regr_post_lbs_N.npy`` that the distributed models do not
carry, and without it the joint regressor is sliced column-wise, which puts the
LBS joints -- what actually drives the fit -- in the wrong place. Renormalising
the sliced rows makes it worse, not better. The full-vertex fit costs 5.4 ms
compiled with a held shape, so there is little to buy and a great deal to lose.

Traps handled here, all from ``docs/smplfitter.md`` and all silent if missed:
``num_betas`` must be given explicitly or the regulariser drives ~400 components
to zero and returns the average body; ``fit()`` ignores a request for vertices,
so the model has to be run forward again; and the hand joints must be relaxed,
since the downstream model has no finger DoF and fitting 45 of them to NLF's
least reliable region mangles them.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from rtcosmik.smpl import torch_shim  # noqa: F401  -- must precede smplfitter

LOGGER = logging.getLogger(__name__)

#: SMPL-X hand joints (15 per hand, after the 25 body/face joints).
HAND_JOINTS = tuple(range(25, 55))

#: Shape components to solve for. Beyond roughly this many, a component changes
#: the surface by less than the input noise and is not identifiable; passing the
#: model's full ~400 makes the default regulariser return the average body.
DEFAULT_NUM_BETAS = 16

GENDERS = {"m": "male", "f": "female", "n": "neutral"}


class SmplRefiner:
    """Replace a dense NLF vertex cloud with the closest plausible SMPL body.

    Call :meth:`refine` once per frame with ``(V, 3)`` vertices in metres, in any
    frame -- the fit is equivariant to rigid motion, so it does not matter
    whether the points are in camera or world coordinates, as long as they are
    consistent.
    """

    def __init__(self, gender="n", num_betas=DEFAULT_NUM_BETAS, num_iter=1,
                 beta_regularizer=1.0, beta_mode="calibrated",
                 calibration_frames=30, calibration_iter=8, warm_start=True,
                 model_name="smplx", device="cuda",
                 zero_hands=True, hand_weight=0.05, model_root=None,
                 compile_online=True, logger=None):
        import smplfitter.pt as smpl_pt
        from smplfitter.pt.bodyfitter import BodyFitter

        if beta_mode not in ("free", "calibrated", "shared"):
            raise ValueError(f"unknown beta_mode {beta_mode!r}")

        self.logger = logger or LOGGER
        self.device = device
        self.beta_mode = beta_mode
        self.calibration_frames = calibration_frames
        self.num_iter = num_iter
        # Calibration happens once per subject, so it can afford iterations the
        # per-frame path cannot: a better shape there is paid for once and then
        # held for the whole trial.
        self.calibration_iter = calibration_iter
        self.warm_start = warm_start
        self.beta_regularizer = beta_regularizer
        self.zero_hands = zero_hands

        gender_name = GENDERS.get(str(gender)[:1].lower(), "neutral")
        root = None if model_root is None else str(Path(model_root) / model_name)
        self.body_model = smpl_pt.BodyModel(model_name, gender_name,
                                            model_root=root,
                                            num_betas=num_betas).to(device)
        self.fitter = BodyFitter(self.body_model).to(device)
        self.num_vertices = self.body_model.num_vertices

        # Hands pull the body toward whatever NLF says about fingers, which is
        # its least reliable region. Down-weighting them is separate from not
        # posing them: one stops them distorting the fit, the other stops the
        # output carrying 45 meaningless DoF.
        self.vertex_weights = None
        if hand_weight is not None and hand_weight != 1.0:
            self.vertex_weights = self._hand_downweights(hand_weight)

        self.betas = None
        self._calibration = []
        self._frames_seen = 0
        self._previous_pose = None
        self.last_residual_mm = float("nan")

        # The fit's cost is per-call overhead rather than the solve, so the
        # online path -- one fused cloud, batch 1 -- is exactly where compiling
        # pays: 3.53 ms to 1.96 ms in the guide's measurement. Compile only the
        # batch-1 shapes; torch.compile recompiles per input shape, and the
        # calibration batch runs once and would cost a second 18 s compile for
        # nothing. CUDA graphs are deliberately not used: capture fails on this
        # code (see docs/smplfitter.md).
        self._fit_one = self.fitter.fit
        self._fit_one_known = self.fitter.fit_with_known_shape
        self._forward_one = self.body_model
        if compile_online:
            self._fit_one = torch.compile(self.fitter.fit, dynamic=False)
            self._fit_one_known = torch.compile(self.fitter.fit_with_known_shape,
                                                dynamic=False)
            self._forward_one = torch.compile(self.body_model, dynamic=False)

    def _hand_downweights(self, weight):
        """Per-vertex weights with the hand vertices reduced.

        Hand vertices are those skinned predominantly to a hand joint.
        """
        skinning = getattr(self.body_model, "weights", None)
        if skinning is None:
            return None
        dominant = torch.as_tensor(skinning).argmax(dim=-1)
        w = torch.ones(self.num_vertices, device=self.device)
        is_hand = torch.zeros_like(dominant, dtype=torch.bool)
        for joint in HAND_JOINTS:
            is_hand |= dominant == joint
        w[is_hand.to(self.device)] = weight
        return w

    def _forward(self, result, compiled=False):
        """Run the body model on fitted parameters; ``fit`` will not return vertices.

        ``fit_with_known_shape`` omits ``shape_betas`` from its result -- it was
        given one -- so the held shape is substituted rather than looked up.
        """
        pose = result["pose_rotvecs"]
        if self.zero_hands:
            pose = pose.clone().reshape(len(pose), -1, 3)
            pose[:, HAND_JOINTS] = 0.0
            pose = pose.reshape(len(pose), -1)
        betas = result.get("shape_betas")
        if betas is None:
            betas = self.betas.expand(len(pose), -1)
        forward = self._forward_one if compiled else self.body_model
        out = forward(pose_rotvecs=pose, shape_betas=betas,
                      trans=result["trans"], return_vertices=True)
        return out["vertices"]

    def _fit(self, target, share_beta, compiled=False):
        keys = ["pose_rotvecs", "shape_betas", "trans"]
        weights = None
        if self.vertex_weights is not None:
            weights = self.vertex_weights.expand(len(target), -1)
        if self.betas is not None:
            fit = self._fit_one_known if compiled else self.fitter.fit_with_known_shape
            # At 40 Hz the body moves millimetres between frames, so the previous
            # pose is a good start and lets a single iteration do the work of
            # several. Always a tensor, never None: torch.compile traces a
            # separate graph per signature, and one graph is the point.
            initial = self._previous_pose
            if initial is None or len(initial) != len(target):
                initial = torch.zeros((len(target), self.body_model.num_joints * 3),
                                      dtype=torch.float32, device=self.device)
            return fit(
                shape_betas=self.betas.expand(len(target), -1),
                target_vertices=target, vertex_weights=weights,
                num_iter=self.num_iter, final_adjust_rots=True,
                initial_pose_rotvecs=(initial if self.warm_start else None),
                requested_keys=keys)
        fit = self._fit_one if compiled else self.fitter.fit
        return fit(
            target, vertex_weights=weights, num_iter=self.num_iter,
            beta_regularizer=self.beta_regularizer, share_beta=share_beta,
            final_adjust_rots=True, requested_keys=keys)

    @torch.inference_mode()
    def warmup(self):
        """Pay the compile up front, like every other engine in this pipeline.

        Both batch-1 graphs are traced: the free fit used before the shape is
        known, and the known-shape fit used after. Doing this lazily instead
        would put an 18 s stall in the middle of a trial and poison the timings.
        """
        if self._fit_one is self.fitter.fit:
            return
        started = time.perf_counter()
        # A real body rather than a synthetic cloud: the fit's control flow
        # depends on the input being something it can actually converge on.
        zeros = torch.zeros((1, self.body_model.num_betas), dtype=torch.float32,
                            device=self.device)
        dummy = self.body_model(
            shape_betas=zeros,
            pose_rotvecs=torch.zeros((1, self.body_model.num_joints * 3),
                                     dtype=torch.float32, device=self.device),
            return_vertices=True)["vertices"]
        saved, self.betas = self.betas, None
        self._forward(self._fit(dummy, share_beta=False, compiled=True),
                      compiled=True)
        self.betas = zeros
        self._previous_pose = None
        self._forward(self._fit(dummy, share_beta=False, compiled=True),
                      compiled=True)
        self.betas = saved
        self._previous_pose = None
        self.logger.info(
            f"SMPL fitter compiled in {time.perf_counter() - started:.1f} s")

    def calibrate(self, frames):
        """Fit one shape over a stack of ``(T, V, 3)`` frames and hold it."""
        target = torch.as_tensor(np.asarray(frames), dtype=torch.float32,
                                 device=self.device)
        with torch.inference_mode():
            result = self.fitter.fit(
                target, num_iter=self.calibration_iter,
                beta_regularizer=self.beta_regularizer, share_beta=True,
                final_adjust_rots=True,
                requested_keys=["pose_rotvecs", "shape_betas", "trans"])
        betas = result["shape_betas"][:1].clone()
        if torch.allclose(betas, torch.zeros_like(betas), atol=1e-6):
            raise RuntimeError(
                "shape_betas came back all zeros: num_betas is too large for the "
                "regulariser (see docs/smplfitter.md, trap 2)")
        self.betas = betas
        self.logger.info(
            f"SMPL shape calibrated on {len(frames)} frames, "
            f"{self.calibration_iter} iters: "
            f"betas[:6] = {np.round(betas[0, :6].cpu().numpy(), 2)}")
        return betas

    def reset(self, beta_mode=None):
        """Forget the fitted shape, keeping the compiled graphs.

        The shape belongs to the subject, so it must not carry across trials --
        but recompiling between trials would cost 18 s each time for nothing.
        """
        self.betas = None
        self._calibration = []
        self._frames_seen = 0
        self._previous_pose = None
        if beta_mode is not None:
            if beta_mode not in ("free", "calibrated", "shared"):
                raise ValueError(f"unknown beta_mode {beta_mode!r}")
            self.beta_mode = beta_mode

    @torch.inference_mode()
    def refine(self, vertices):
        """One frame in, one frame out: ``(V, 3)`` metres to ``(V, 3)`` metres."""
        target = torch.as_tensor(np.asarray(vertices)[None], dtype=torch.float32,
                                 device=self.device)
        self._frames_seen += 1

        if self.beta_mode == "calibrated" and self.betas is None:
            # Collect a window, then switch to the fixed shape. Until it is
            # ready every frame is solved free, which keeps this causal.
            self._calibration.append(np.asarray(vertices))
            if len(self._calibration) >= self.calibration_frames:
                stack = self._calibration
                self._calibration = []
                self.calibrate(stack)

        result = self._fit(target, share_beta=False, compiled=True)
        if self.warm_start:
            self._previous_pose = result["pose_rotvecs"].detach()
        fitted = self._forward(result, compiled=True)
        self.last_residual_mm = float(
            (fitted - target).norm(dim=-1).median() * 1000)
        return fitted[0].cpu().numpy()

    @torch.inference_mode()
    def refine_batch(self, frames, share_beta=True):
        """Offline path: fit a whole ``(T, V, 3)`` stack at once.

        Batching across time is what makes the fitter cheap -- per-call overhead
        dominates the solve -- so this is far faster per frame than
        :meth:`refine`. It is not causal and exists to bound what the online
        modes give up.
        """
        target = torch.as_tensor(np.asarray(frames), dtype=torch.float32,
                                 device=self.device)
        result = self.fitter.fit(
            target, vertex_weights=(None if self.vertex_weights is None
                                    else self.vertex_weights.expand(len(target), -1)),
            num_iter=self.num_iter, beta_regularizer=self.beta_regularizer,
            share_beta=share_beta, final_adjust_rots=True,
            requested_keys=["pose_rotvecs", "shape_betas", "trans"])
        fitted = self._forward(result)
        self.last_residual_mm = float(
            (fitted - target).norm(dim=-1).median() * 1000)
        return fitted.cpu().numpy()
