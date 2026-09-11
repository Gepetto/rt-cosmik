# smplfitter — a standalone working guide

Self-contained. Nothing here needs RT-COSMIK, NLF, or any particular codebase —
the runnable example builds its own input. A final section covers feeding it from
a dense-vertex regressor such as NLF, clearly marked as optional.

Every trap below was hit in a real session. None of them announce themselves:
three fail silently and hand you a plausible-looking wrong answer.

---

## 0. What smplfitter does

Given **3D points that correspond to SMPL body-model vertices** (vertex *i* of your
input is vertex *i* of SMPL), it recovers the body-model parameters that explain
them: shape `betas`, pose `pose_rotvecs`, and translation `trans`.

It is *not* a pose estimator from images and *not* a registration method — the
correspondence must already exist. That is what makes it fast.

Input `(T, V, 3)` where `V` is 6890 (SMPL / SMPL-H) or 10475 (SMPL-X).

---

## 1. Install

```bash
pip install smplfitter
```

### Body models (registration required, interactive)

Register with the **same email and password** at all three, then run the
downloader:

- https://smpl.is.tue.mpg.de/ (SMPL)
- https://smpl-x.is.tue.mpg.de/ (SMPL-X)
- https://mano.is.tue.mpg.de/ (MANO and SMPL+H)

```bash
python3 -m smplfitter.download /path/to/body_models
export SMPLFITTER_BODY_MODELS=/path/to/body_models
```

The downloader **prompts for credentials**, so it cannot run in a Dockerfile
build step or any non-interactive context. Download once on a machine with a
terminal and mount or copy the directory.

Layout produced:

```
body_models/
  smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz
  smplh/SMPLH_{male,female}.pkl
  smpl/ ...
```

`SMPLFITTER_BODY_MODELS` is read at import time. Alternatives: `DATA_ROOT`
pointing at the parent of `body_models/`, or pass `model_root=` explicitly.

---

## 2. Trap 1 — `torch.nn.Buffer`, fails loudly on torch < 2.5

smplfitter 0.5 uses `nn.Buffer`, added in **torch 2.5**:

```
AttributeError: module 'torch.nn' has no attribute 'Buffer'
```

If you cannot move torch (TensorRT engines and torchscript models are built
against a specific version), shim it. Save as `torch_shim.py` and import it
**before** smplfitter:

```python
# torch_shim.py
import torch, torch.nn as nn

if not hasattr(nn, "Buffer"):
    class Buffer(torch.Tensor):
        def __new__(cls, data=None, *, persistent=True):
            if data is None:
                data = torch.empty(0)
            t = torch.Tensor._make_subclass(cls, data, data.requires_grad)
            t._persistent = persistent
            return t

    _orig = nn.Module.__setattr__

    def _setattr(self, name, value):
        if isinstance(value, Buffer):
            plain = value.as_subclass(torch.Tensor)
            if "_buffers" in self.__dict__:
                for d in (self.__dict__, self.__dict__.get("_parameters", {})):
                    d.pop(name, None)
                self.register_buffer(name, plain,
                                     persistent=getattr(value, "_persistent", True))
                return
            value = plain
        _orig(self, name, value)

    nn.Buffer = Buffer
    nn.Module.__setattr__ = _setattr
```

Verified with this shim: `.to("cuda")` still moves buffers correctly, and fitted
results match.

---

## 3. Trap 2 — `num_betas`, the silent one

The SMPL-X `.npz` exposes roughly **400 shape components**. `get_cached_body_model()`
takes all of them, and the default `beta_regularizer=1.0` then drives every one
to zero. The fit reports success and returns **the average body**.

Always construct with an explicit `num_betas`:

```python
import smplfitter.pt as P
bm = P.BodyModel("smplx", "female", num_betas=16)     # NOT get_cached_body_model()
```

Check the result is not degenerate:

```python
print(np.round(betas[0], 2))
# healthy:   [ 0.87 -1.25  0.20 -0.07 ...]
# degenerate:[ 0.   -0.    0.    0.   ...]   <- num_betas too large
```

10–16 betas is the usual working range.

---

## 4. Trap 3 — `fit()` does not return vertices

`requested_keys=[..., "vertices"]` is **silently ignored**. Run the body model
forward on the fitted parameters:

```python
res = fitter.fit(tv, num_iter=4, beta_regularizer=1.0, share_beta=True,
                 final_adjust_rots=True,
                 requested_keys=["pose_rotvecs", "shape_betas", "trans"])

fwd = bm(pose_rotvecs=res["pose_rotvecs"],
         shape_betas=res["shape_betas"],
         trans=res["trans"], return_vertices=True)
V = fwd["vertices"]     # (T, V, 3)
J = fwd["joints"]       # (T, 55, 3) for SMPL-X
```

---

## 5. Runnable example — no external data needed

Generates a body, perturbs it with noise, fits it back, and reports the recovery
error. Use this to verify an installation end to end.

```python
import torch_shim                       # must come first (section 2)
import numpy as np, torch
import smplfitter.pt as P
from smplfitter.pt.bodyfitter import BodyFitter

NB, DEV, T = 16, "cuda" if torch.cuda.is_available() else "cpu", 8
bm = P.BodyModel("smplx", "neutral", num_betas=NB).to(DEV)
fitter = BodyFitter(bm).to(DEV)

# ground truth: one shape, several poses.
# NOTE pose_rotvecs is the FULL pose, num_joints*3 (165 for SMPL-X: 55 joints).
# There is no `body_pose` argument -- that is the smplx package's API, not this one.
rng = np.random.default_rng(0)
NJ = bm.num_joints
# realistic shape: low-order betas dominate, as real bodies do
beta_true = rng.normal(size=NB) * np.exp(-np.arange(NB) / 4)
pose_true = np.zeros((T, NJ * 3))
pose_true[:, 3:66] = rng.normal(size=(T, 63)) * 0.25      # body joints only
gt = bm(shape_betas=torch.tensor(np.repeat(beta_true[None], T, 0),
                                 dtype=torch.float32, device=DEV),
        pose_rotvecs=torch.tensor(pose_true, dtype=torch.float32, device=DEV),
        return_vertices=True)
V_gt = gt["vertices"]

# observations: ground truth + 1 cm noise
V_obs = V_gt + torch.randn_like(V_gt) * 0.01

res = fitter.fit(V_obs, num_iter=4, beta_regularizer=0.1, share_beta=True,
                 final_adjust_rots=True,
                 requested_keys=["pose_rotvecs", "shape_betas", "trans"])
fwd = bm(pose_rotvecs=res["pose_rotvecs"], shape_betas=res["shape_betas"],
         trans=res["trans"], return_vertices=True)

err_noisy = (V_obs - V_gt).norm(dim=-1).median().item() * 1000
err_fit   = (fwd["vertices"] - V_gt).norm(dim=-1).median().item() * 1000
print(f"input noise {err_noisy:.1f} mm -> after fit {err_fit:.1f} mm")
print("betas recovered:", np.round(res['shape_betas'][0].detach().cpu().numpy()[:6], 2))
print("betas true     :", np.round(beta_true[:6], 2))
```

Expected output on a healthy install:

```
input noise 15.4 mm -> after fit 6.4 mm
betas recovered: [ 0.15 -0.43  0.39  0.34  0.21 -0.46]
betas true     : [ 0.13 -0.1   0.39  0.05 -0.2   0.1 ]
```

Two things to read from it. **The fit roughly halves the vertex error** — that is
the denoising you are buying. And the **leading betas recover closely while
high-order ones do not**: components beyond the first few change the surface by
less than the input noise, so they are simply not identifiable. That is a
property of the problem, not a bug, and no amount of lowering
`beta_regularizer` fixes it (measured: L2 beta error 3.64 at reg=1.0, still 1.96
at reg=0.03, against a true magnitude of 3.82).

If the betas come back **all zeros**, that is Trap 2 — `num_betas` is too large.

---

## 6. Arguments that matter

| argument | effect |
|---|---|
| `share_beta=True` | one shape across the whole batch, poses independent. What you want when the batch is one subject. |
| `num_iter` | 1–4. See timings; 4 is the practical online maximum. |
| `beta_regularizer` | 1.0 is fine **once `num_betas` is sane**. |
| `vertex_weights` | `(T, V)`. Down-weight unreliable vertices; set 0 to ignore them. |
| `initial_shape_betas` | warm start from a known shape. **Does not reduce runtime** — see below. |
| `final_adjust_rots` | leave on. |

---

## 7. Timings (RTX 4500 Ada, SMPL-X, 16 betas)

```
1 frame,  4 iters, beta free                 9.5 ms
1 frame,  4 iters, beta given                10.5 ms    <- a known beta does NOT help
1 frame,  2 iters, beta given                 6.6 ms
1 frame,  1 iter,  beta given                 4.6 ms
8 frames batched,  1 iter                     1.48 ms/frame
12 frames batched, 2 iters                    1.61 ms/frame
200 frames batched, 2 iters                   0.76 ms/frame
```

**The cost is per-call overhead, not the solve.** That is why fixing `betas` buys
nothing and why batching buys a great deal.

### Cutting the overhead

`torch.compile` removes most of it, safely:

```python
fit_fn = torch.compile(fitter.fit, dynamic=False)   # ~18 s one-off compile
res = fit_fn(tv, num_iter=1, beta_regularizer=1.0, share_beta=True,
             final_adjust_rots=True,
             requested_keys=["pose_rotvecs", "shape_betas", "trans"])
```

```
batch 1, 1 iter, eager                       3.53 ms
batch 1, 1 iter, torch.compile               1.96 ms      <- 1.8x
batch 4, 1 iter, eager                      12.22 ms
batch 4, 1 iter, torch.compile               9.06 ms
```

Verified equivalent: compiled vs eager differ by **0.0003 mm median / 0.029 mm
max on the vertices**, 24,000x smaller than the fit's own 7.2 mm residual. Note
it recompiles per input shape, so pin your batch size.

CUDA graph capture **fails** on this code (`operation failed due to a previous
error during capture`) — do not spend time on it.

### Batching across cameras does NOT help

If you have several calibrated views of one instant, fuse the 3D clouds first and
fit once. Measured:

```
                                        cost      rigid-distance scatter
raw fused cloud, no fit                   --              0.42 mm
fuse then fit           (batch 1)      4.80 ms            0.06 mm
fit each camera, average (batch 4)    12.22 ms            0.06 mm
```

2.5x the cost for identical accuracy — the multiview information is already in
the fused cloud. The cost step is from batch 1 to 2 (4.8 -> 11.8 ms); from 2 to 8
it is nearly flat (12.8 ms), so batching across *time* is what pays, at the price
of latency.

---

## 8. Hands come out broken, and it is not SMPL's fault

SMPL-X carries 15 articulated joints per hand (MANO), 45 DoF. A fit drives all of
them from whatever your input says about fingers, which is usually its least
reliable region. The fingers come out mangled.

If your downstream model has no finger DoF, do not pose them:

```python
HAND_J = list(range(25, 55))                       # SMPL-X hand joints
pose = res["pose_rotvecs"].clone().reshape(T, -1, 3)
pose[:, HAND_J] = 0.0                              # relaxed hands
fwd = bm(pose_rotvecs=pose.reshape(T, -1), shape_betas=res["shape_betas"],
         trans=res["trans"], return_vertices=True)
```

Optionally also stop the hands pulling the body: `vertex_weights[:, hand_ids] = 0.05`.

Measured effect on a real fit: hand distortion **7.2% → 0.0%**, body fit unchanged
(6.5 → 6.4 mm).

---

## 9. Optional — feeding it from NLF

NLF's canonical vertices **are** SMPL vertices: its `canonical_verts/smplx.npy` is
(10475, 3) and index *i* is SMPL-X vertex *i*. So NLF's dense output is a valid
smplfitter input with no correspondence step.

Ask NLF for all vertices rather than a marker subset:

```python
weights = nlf_model.get_weights_for_canonical_points(
    torch.from_numpy(np.load("canonical_verts/smplx.npy")).float().cuda())
out = nlf_model.estimate_poses_batched(imgs, boxes, intrinsic_matrix=K,
                                       weights=weights, num_aug=1)
```

Measured: **16 ms/frame for all 10475 vertices across 4 cameras** — the same as
for 43 markers, because the cost is the backbone, not the output head.

### What the fit buys on real data

Measured against a marker-based mocap system, 200 frames, 4 cameras. Scatter of
distances that must be rigid:

```
                      mocap (floor)   raw regressed   after SMPL fit
mean scatter              1.2 mm          5.3 mm          3.3 mm
mean |length error|         --            2.6 cm          2.2 cm
shank length            42.2 cm         45.5 cm         43.4 cm
```

The fit closes about half the gap to a lab mocap system, and corrected a 3 cm
systematic shank-length error. Frame-to-frame motion is unchanged (5.7 mm both
ways), so it denoises **shape** without smoothing over time — no lag.

It also halves systematic surface distortion (12.0% → 5.5% of vertices deviating
more than 30% in local edge length), while sitting exactly at the reprojection
noise floor (12.77 px fitted vs 12.87 px raw).

Known exception: **shoulder width gets worse** (scatter 12.2 → 13.6 mm, mean
28.8 → 30.7 cm against a 28.6 cm truth). SMPL's shoulder girdle is its weakest
region.

---

## 10. Unrelated gotchas worth knowing

- **`import coacd` next to `import pinocchio` segfaults** — a C++ library
  conflict. Run convex decomposition in a separate process.
- **VolumetricSMPL needs `pytorch3d`** at inference, not only for training. The
  prebuilt wheel index works when torch/CUDA match exactly:
  `pip install --no-deps pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html`
- **SMPL-X is three connected components** — the body plus two eyeball shells.
  Anything expecting one watertight solid must drop the eyeballs first.
