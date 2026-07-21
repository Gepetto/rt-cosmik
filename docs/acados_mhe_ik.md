# Parameterizing the acados MHE-IK to compile ONCE (design & build guide)

**The task this document is for:** make the acados MHE-IK solver **subject-independent**
so it is code-generated + compiled **exactly once** (ship the `.so` in the
container) and every individual — even in the live pipeline — is handled by just
**setting runtime parameters**, never recompiling.

Today the acados backend bakes the human's forward-kinematics into the generated C,
so it recompiles ~30–40 s for every new calibrated model. The whole point here is
to remove that.

This guide is self-contained: minimal context, then the concrete plan, code
skeletons, the acados parameter API, validation, and gotchas. Feasibility has been
**verified** (see §3).

---

## 1. Context you need (short)

- Two interchangeable MHE-IK smoothers live in
  [`src/rtcosmik/ik/ik.py`](../src/rtcosmik/ik/ik.py): `RT_SWIKA_FATROP` (validated
  reference) and `RT_SWIKA_ACADOS` (fast, reproduces the same OCP). Both share:
  ```python
  X, U = solver.solve(X, U, marker_meas, X0, cost_weights, dt)   # q = X[:nq, -1]
  ```
- The OCP (nodes `k=0..N-1`, `x=[q;dq]`, `u=ddq`): Euler dynamics
  `q⁺=integrate(q,dq·dt)`, `dq⁺=dq+u·dt`; cost
  `Σ w0‖markers(q_k)−meas_k‖² + w1‖x_k−X0‖² + w2‖u_k‖²`; bounds `lower≤q_k[7:]≤upper`;
  soft arrival cost (no hard x0 clamp); output = q at the newest node.
- **Where the per-subject dependence lives:** the term `markers(q)` — the marker
  forward-kinematics — is built in `RT_SWIKA_ACADOS._build_marker_fk_expr` via
  `cpin.framesForwardKinematics`, using the calibrated model's **numeric**
  `jointPlacements` and marker-frame `placement`s. Those numbers differ per person,
  so they get baked into the generated C → recompile per person.
- `dt` is also baked (into `integrate(q, dq*dt)`); keep it fixed (it is — `1/fs`).
- To run acados you need `ACADOS_SOURCE_DIR` set and its libs on the loader path,
  and acados_template installed **`--no-deps`** (a pip `casadi` wheel breaks
  `pinocchio.casadi` — see [full env notes] below in §8).

Model build (per subject, unchanged and cheap — this stays):
`HumanLoader → scale_human_model → mks_registration → RT_IK(IPOPT) init → recalibrate_marker_frames_in_joint_space`
(`src/rtcosmik/human_model/model_utils.py`). Only the acados **compile** is what we
eliminate; scale/register/recalibrate still run per subject to produce the numbers
we'll now inject as parameters.

---

## 2. What varies per individual (this is the whole crux)

Only **geometry translations** change from person to person:

| what | where in the `pin.Model` | set by |
|---|---|---|
| **segment lengths** | `jointPlacements[j].translation` (3 each) | `scale_human_model` |
| **marker offsets** | tracked-frame `.placement.translation` (3 each) | `recalibrate_marker_frames_in_joint_space` |

**Everything else is shared by everyone and stays baked:** kinematic topology, joint
types & axes, joint-placement **rotations**, marker-frame **rotations** (identity),
the freeflyer's enforced orientation, joint **limits** (angular ranges don't change
with limb length), `N`, `dt`, the cost/constraint structure. Cost weights are
already set at runtime. So the parameter vector is just a stack of 3-vectors.

Rough size: ~ (internal joints) × 3 + (tracked keys) × 3 ≈ 120 + 87 ≈ **~200 scalars**
for the current 43-dof model / 29 tracked markers. Parameters are constants (not
optimization variables) → **zero extra solve cost**.

---

## 3. Feasibility — VERIFIED

`pinocchio.casadi` will carry symbolic placements through FK. Confirmed this session
(both a segment length and a marker offset propagate, and the FK stays a function of
`q`):

```python
import numpy as np, casadi, pinocchio.casadi as cpin
cm = cpin.Model(model); q = casadi.SX.sym('q', model.nq)

# (a) segment length -> parameter.  BOTH args to cpin.SE3 must be casadi types.
L = casadi.SX.sym('L', 3)
R = casadi.SX(np.array(model.jointPlacements[jid].rotation))   # numpy -> SX (required!)
cm.jointPlacements[jid] = cpin.SE3(R, L)                       # oMi[jid] now depends on L  ✓

# (b) marker offset -> parameter.  Use get-modify-set on the frame.
off = casadi.SX.sym('off', 3)
fr = cm.frames[fid]
fr.placement = cpin.SE3(casadi.SX(np.array(fr.placement.rotation)), off)
cm.frames[fid] = fr                                            # oMf[fid] now depends on off ✓

cpin.framesForwardKinematics(cm, cm.createData(), q)           # markers(q, [L.., off..]) symbolic
```

Two facts that make this clean:
- `cpin.integrate(cm, q, v)` does **not** depend on placements (it's on the config
  manifold) → the **dynamics need no parameters**. Only `markers(q,p)` (the cost)
  does.
- Passing a **numpy** rotation to `cpin.SE3` raises a Boost.Python `ArgumentError`;
  always wrap as `casadi.SX(np.array(R))`.

---

## 4. Implementation plan

Recommended shape: a new class **`RT_SWIKA_ACADOS_PARAM`** next to `RT_SWIKA_ACADOS`
(keep the latter for A/B comparison), plus a `settings.mhe_backend = "acados_param"`.
Model it closely on `RT_SWIKA_ACADOS` — the only differences are (i) symbolic
placements, (ii) `model.p`, (iii) `parameter_values`, (iv) a `set_model_params()`
method, and (v) `build` is one-time-global (not per subject).

### 4.1 Build the parameterized model + solver (ONCE)

Give the constructor a **structural** `pin.Model` that already has the marker frames
registered (any calibrated model works — its numbers are just the *defaults*; the
structure, rotations, limits are what matter and are shared).

```python
def _build_parameterized(self, pin_model, keys):
    cm = cpin.Model(pin_model)
    params, defaults = [], []
    self._joint_ids, self._frame_ids = [], []

    # (a) internal joint placements (skip universe id 0 and the freeflyer)
    for jid in range(1, pin_model.njoints):
        if pin_model.joints[jid].nq == 7:      # freeflyer -> keep baked (enforced FF orientation)
            continue
        L = casadi.SX.sym(f"L_{jid}", 3)
        R = casadi.SX(np.array(pin_model.jointPlacements[jid].rotation))
        cm.jointPlacements[jid] = cpin.SE3(R, L)
        params.append(L); defaults.append(np.array(pin_model.jointPlacements[jid].translation))
        self._joint_ids.append(jid)

    # (b) tracked marker frame offsets
    for key in keys:
        fid = pin_model.getFrameId(key)
        off = casadi.SX.sym(f"off_{fid}", 3)
        fr = cm.frames[fid]
        fr.placement = cpin.SE3(casadi.SX(np.array(fr.placement.rotation)), off)
        cm.frames[fid] = fr
        params.append(off); defaults.append(np.array(pin_model.frames[fid].placement.translation))
        self._frame_ids.append(fid)

    self._p = casadi.vertcat(*params)
    self._p_default = np.concatenate(defaults)       # order MUST match extraction (§4.3)
    self._cm = cm
    return cm
```

Then the FK / acados wiring (mirror `RT_SWIKA_ACADOS._create_ocp_solver`):
```python
cpin.framesForwardKinematics(cm, cm.createData(), cq)          # cq = cx[:nq]
markers_expr = casadi.vertcat(*[cm.data.oMf[fid].translation for fid in tracked_fids])
model.x, model.u, model.p = cx, cu, self._p
model.disc_dyn_expr  = vertcat(cpin.integrate(cm, cq, cdq*dt), cdq + cu*dt)   # no p dependence
model.cost_y_expr    = casadi.vertcat(markers_expr, cx, cu)     # depends on p
model.cost_y_expr_e  = casadi.vertcat(markers_expr, cx)
model.con_h_expr     = cx[7:nq]                                 # no p dependence
...
ocp.parameter_values = self._p_default                          # sets np; REQUIRED
solver = AcadosOcpSolver(ocp, json_file=..., generate=build, build=build)
```
Everything else (W/W_e blocks, yref layout, `PARTIAL_CONDENSING_HPIPM`,
`GAUSS_NEWTON`, `SQP`, `nlp_solver_max_iter`, `con_h` limits, `N_horizon=N-1`, the
`solve()` window mapping and warm start) is **identical** to `RT_SWIKA_ACADOS`.

### 4.2 Set parameters per subject (no recompile)

```python
def set_model_params(self, calibrated_model):
    vals = [np.array(calibrated_model.jointPlacements[j].translation) for j in self._joint_ids]
    vals += [np.array(calibrated_model.frames[f].placement.translation) for f in self._frame_ids]
    p = np.concatenate(vals)                       # SAME order as _p_default
    for k in range(self._N):                       # geometry is constant over the horizon
        self._ocp_solver.set(k, "p", p)
```
Call this once after each subject's calibration; then `solve()` as usual. (If your
acados version exposes global parameters — `set_p_global_and_precompute_dependencies`
— you can set once instead of per-stage; per-stage is the safe default.)

### 4.3 Pipeline / benchmark integration
- Build `RT_SWIKA_ACADOS_PARAM` **once** at startup (or ship a prebuilt export dir
  and construct with `build=False`).
- After the per-subject `scale→register→IPOPT→recalibrate`, call
  `set_model_params(calibrated_model)` instead of reconstructing the solver.
- Structure identical across subjects (same URDF + same `MKS_COSMIK_2_JOINTS`),
  so the joint/frame id lists are stable.

---

## 5. Validation (do this FIRST, before wiring anything)

**5a. FK regression** — the parameterized FK with the calibrated `p` must equal the
baked FK of the current per-subject model, to ~1e-9:
```python
f_baked = casadi.Function('fb', [cq], [markers_expr_from_baked_calibrated_model])
f_param = casadi.Function('fp', [cq, p_sym], [markers_expr_parameterized])
p_cal   = extract_params(calibrated_model)     # via the §4.2 ordering
for _ in range(200):
    qr = pin.randomConfiguration(model)
    assert np.allclose(np.array(f_baked(qr)), np.array(f_param(qr, p_cal)), atol=1e-9)
```
This catches any parameter-ordering or convention mistake immediately.

**5b. End-to-end A/B** — in the benchmark, add `RT_SWIKA_ACADOS_PARAM` as a third
backend; on the same data its per-frame `q` and RMSE must match `RT_SWIKA_ACADOS`
(the baked one) to solver tolerance, and the "reuse" is now automatic (no compile
after the first). Reuse the existing harness in
[`tests/benchmark/benchmark_mhe_ik_backends.py`](../tests/benchmark/benchmark_mhe_ik_backends.py).

**5c. Cross-subject** — build once from subject A's structure, then
`set_model_params(subject_B_model)` and confirm markers/RMSE match a freshly-baked
subject-B solver. This is the actual win: one `.so`, many people.

---

## 6. Gotchas & risks (read before coding)

- **`cpin.SE3(R, t)` casadi typing:** both args must be casadi (`casadi.SX(np.array(R))`).
- **Parameter ordering** in `_p_default` (build) and `set_model_params` (runtime)
  **must be byte-for-byte the same**. The FK regression (5a) is your guard.
- **Skip the freeflyer** joint placement (its rotation is the enforced FF
  orientation; translation is 0 and constant). Parameterize internal joints only.
- **Only parameterize tracked-key frames** (only they enter the cost). Other
  registered marker frames can stay baked.
- **Joint limits** (`con_h` bounds `lower/upperPositionLimit[7:]`) are angular →
  subject-independent → keep baked. Confirm `scale_human_model`/`mks_registration`
  never touch position limits (they only set placements/add frames).
- **`integrate` must not gain a `p` dependence.** Verify with `casadi.depends_on`.
- **acados parameter API** differs slightly across versions — verify
  `ocp.parameter_values` sets `np` and `ocp_solver.set(stage,'p',...)` works on the
  pinned build (commit `8e1a6f856`).
- **Codegen size:** ~200 params in the FK expression makes the generated C a bit
  larger and codegen a touch slower — **one-time only**; solve time is unchanged
  (constants, no new variables). Validate this claim on the real model.
- The `scale→register→IPOPT→recalibrate` calibration **still runs per subject**
  (seconds) to produce `p`; only the ~30–40 s acados compile is removed.

**Effort:** ~1–2 focused days including the three validation stages. **Payoff:**
compile once, ship the `.so`; every subject and every live session just sets `p`.
This makes the benchmark hash-cache and the per-run pipeline recompile obsolete.

---

## 7. Reference: how the current baked solver is built

To mirror conventions, read `RT_SWIKA_ACADOS` in
[`src/rtcosmik/ik/ik.py`](../src/rtcosmik/ik/ik.py):
- `__init__` → `_create_ocp_solver(build)`; `_build_marker_fk_expr(cq)` (the FK to
  parameterize); `_build_block_weight(...)` (W/W_e); `solve(...)` (window→yref
  mapping, warm start, read-back). Constructor already takes `max_iter`, `build`,
  `export_dir`, `acados_source_dir` — reuse those.
- Model build to feed it: `build_model()` in
  [`tests/benchmark/benchmark_mhe_ik_backends.py`](../tests/benchmark/benchmark_mhe_ik_backends.py)
  (HumanLoader → scale → register → IPOPT init → recalibrate). `nq=43, nv=42`,
  29 tracked keys.

---

## 8. Env & operational notes (only what you need to run/compile acados)

- **acados** built from source, pinned commit `8e1a6f856` (v0.5.4-20), installed to
  `<acados>` with `-DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_AUTOMATIC`. Needs
  `ACADOS_SOURCE_DIR=<acados>`, its `lib/*.so` on the loader path, and the
  `t_renderer` binary at `<acados>/bin/t_renderer` (tera v0.2.0, downloadable from
  the tera_renderer releases). CPU-only (HPIPM) — no GPU needed for IK.
- **CASADI PITFALL:** `pip install acados_template` pulls a pip `casadi` wheel that
  shadows the from-source casadi and **breaks `import pinocchio.casadi`**. Install
  with `--no-deps` (+ `matplotlib cython Deprecated` separately). If broken:
  `pip uninstall -y casadi`. The dev-container Dockerfile
  (`cosmik-dev-container/.devcontainer/Dockerfile`, has the acados stage) does this
  correctly; it also RAM-limits the from-source builds (`build_jobs`) to avoid OOM.
- **Timing hygiene when benchmarking:** governor `performance`, measure without
  `--display`, compare **medians** (acados `sqp_iter` and thus solve time are
  data-dependent; cap with `settings.mhe_max_iter=2..3` for bounded real-time).

### Reference numbers (i7-8850H, N=10, 272 frames)
fatrop(python) ~31 ms / 9.93 mm · acados(default) ~10 ms / 4.29 mm (spikes to
~200 ms on hard frames) · acados(max_iter=3) ~5 ms / ~3.7 mm · agreement mean
`|Δq|` 1.3e-2 rad · first acados build ~35 s, cached reuse ~0.2 s.

### Known pre-existing bugs (unrelated, in the `mhe` branch)
- `pipeline.py` online `mhe` steady-state uses bare `settings.ik_type` (should be
  `self.settings.ik_type`); and references an undefined `viz_human`. Never hit
  because default `ik_type='sbs'`. `run_pipeline.py` (offline) is fine.
