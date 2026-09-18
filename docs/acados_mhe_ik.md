# Parameterizing the acados MHE-IK to compile ONCE

**Goal:** make the acados MHE-IK solver subject-independent so it is code-generated
and compiled **exactly once** (ship the `.so` in the container), and every
individual — including in the live pipeline — is handled by **setting runtime
parameters**. No recompile, ever.

Today `RT_SWIKA_ACADOS` bakes the human's forward kinematics into the generated C,
so it recompiles for every newly calibrated model. In the online pipeline that is a
20–40 s stall the moment a new person steps in front of the cameras.

Everything in §1–§3 has been **measured on the real 43-dof model**, not assumed.

---

## 1. Verified facts

### 1.1 Only translations vary between people

Two raw `HumanLoader` models (1.64 m / 51 kg / female vs 1.87 m / 90 kg / male):

| quantity | max abs difference |
|---|---|
| `jointPlacements[j].rotation` | **0.0** |
| `frames[i].placement.rotation` | **0.0** |
| `lowerPositionLimit` / `upperPositionLimit` | **0.0** |
| `nq`, `nv`, `njoints`, `nframes`, joint names, joint types, frame names | identical |
| `jointPlacements[j].translation` | 7.0e-2 (16 of 37 joints) |
| `inertias[j].mass` | 7.2 (irrelevant — see below) |

Inertias differ but never enter this OCP: the cost is pure forward kinematics and
the dynamics is Euler integration on the configuration manifold. No mass property
is referenced anywhere in `_create_ocp_solver`.

The two calibration steps confirm this by construction:
- `scale_human_model` writes **only** `model.jointPlacements[jid].translation`.
- `recalibrate_marker_frames_in_joint_space` writes **only**
  `model.frames[fid].placement.translation` (its one `rotation` mention is a read,
  `oMj.rotation.T @ (...)`).
- `mks_registration` adds frames with `pin.SE3(np.eye(3), trans)` — identity
  rotation, and it iterates the **hardcoded** `SGTS_MKS_MAPPING`, not the marker
  dict, so the frame set and its order are the same for everyone or the build
  raises. That determinism is what makes frame ids reusable across subjects.

### 1.2 The symbolic parameterization works

- `cpin.integrate(cm, q, v)` does **not** depend on placements — confirmed with
  `casadi.depends_on(...) == False`. **The dynamics needs no parameters.** Only the
  cost does.
- A joint-placement parameter propagates into `oMi` and into every downstream
  `oMf` — confirmed `True`.
- A marker-frame offset parameter propagates into its `oMf` — confirmed `True`.

### 1.3 It reproduces the baked model exactly

On the real calibrated model (`nq=43`, `nv=42`, 38 joints, 118 frames, 29 tracked
markers), parameter vector **195 scalars** = 36 internal joints × 3 + 29 marker
offsets × 3:

```
FK regression, 200 random q, parameterized-at-default vs baked
    max |diff| = 0.000e+00

CROSS-SUBJECT: subject A's parameterized FK fed subject B's p,
               vs subject B's own baked FK, 200 random q
    max |diff| = 0.000e+00        <- this is the whole win
```

(`p` differs between those two subjects by up to 63.1 mm, so this is not a
degenerate comparison.)

### 1.4 It costs essentially nothing to solve

| | baked | parameterized |
|---|---|---|
| codegen + compile | 20.4 s | 32.1 s — **once, ever** |
| export directory | 7.3 MB | 8.1 MB |
| **solve time (median / p95)** | 0.59 / 0.59 ms | **0.60 / 0.60 ms (+2.2%)** |
| `set(k, "p", ...)` over all stages | — | 0.036 ms, once per subject |

Parameters are constants, not decision variables, so there is no extra QP work.
The +2.2% is function-evaluation overhead and is within run-to-run noise.

---

## 2. Two gotchas that will bite

**The rotation must come from the NUMERIC model.** `cm.frames[fid].placement.rotation`
is *already* a CasADi `SX` once `cm` is a `cpin.Model`, and `np.array()` on it raises
`Implicit conversion of symbolic CasADi type to numeric matrix not supported`. Read
rotations from the `pin.Model`, never from the `cpin.Model`:

```python
fr = cm.frames[fid]
fr.placement = cpin.SE3(casadi.SX(np.array(model.frames[fid].placement.rotation)), off)
cm.frames[fid] = fr          # get-modify-set; in-place mutation does not stick
```

Both arguments to `cpin.SE3` must be CasADi types — passing a numpy rotation raises
a Boost.Python `ArgumentError`.

**A propagation test must sample frames that are actually downstream.** Checking
`depends_on(oMf[some_frame], L_j)` for a frame that is not a descendant of joint `j`
returns `False` for a perfectly correct parameterization. Test each joint against a
frame known to be below it, or test the tracked-marker FK as a whole.

---

## 3. Design

A new class `RT_SWIKA_ACADOS_PARAM` beside `RT_SWIKA_ACADOS` (keep the latter for
A/B), selected by `settings.mhe_backend = "acados_param"`. Differences from the
baked class are confined to: symbolic placements, `model.p`, `ocp.parameter_values`,
a `set_model_params()` method, a structural fingerprint, and `build` becoming a
one-time global concern rather than a per-subject one.

Everything else is unchanged and must stay bit-identical: `N_horizon = N-1`, the
`W`/`W_e` block layout, yref mapping, `PARTIAL_CONDENSING_HPIPM`, `GAUSS_NEWTON`,
`SQP`, `DISCRETE`, the soft arrival cost with no `constraints.x0`, `con_h` limits,
warm start and read-back.

### 3.1 Parameter vector

Order is **load-bearing** — build and runtime must agree scalar for scalar:

1. for `jid` in `1 .. njoints-1`, skipping the freeflyer (`joints[jid].nq == 7`):
   `jointPlacements[jid].translation` (3)
2. for `key` in `keys_to_track` **in list order**:
   `frames[getFrameId(key)].placement.translation` (3)

Skip the freeflyer deliberately: its translation is a constant zero and its rotation
is the enforced freeflyer orientation. Parameterize only the tracked marker frames —
the other ~50 registered frames never enter the cost.

### 3.2 Stage parameters, not `p_global`

Use `model.p` + `ocp.parameter_values` + `ocp_solver.set(k, "p", ...)` for every
stage `k in 0..N-1`. This build does expose `p_global` and
`set_p_global_and_precompute_dependencies`, which is arguably the better semantic
fit for something constant across stages — but stage `p` is what §1.4 measured, and
the cost it is meant to save is 0.036 ms per subject. Treat `p_global` as an
optional refinement, not part of the first implementation.

### 3.3 The structural fingerprint — do not skip this

The dangerous failure is **silent misalignment**: if `settings.marker_names`,
`keys_to_track_list`, `SGTS_MKS_MAPPING` or the URDF ever change, the shipped `.so`
still loads and still solves, but `p` now means something different and the FK is
quietly wrong. There is no exception to catch.

Guard it. At build, write a fingerprint next to the generated code covering
everything baked into the `.so`:

```python
fingerprint = hashlib.sha256(json.dumps({
    "joint_names":  list(model.names),
    "joint_types":  [model.joints[j].shortname() for j in range(model.njoints)],
    "frame_names":  [f.name for f in model.frames],
    "keys_to_track": list(keys_to_track),
    "param_joint_ids": self._joint_ids,
    "param_frame_ids": self._frame_ids,
    "nq": model.nq, "nv": model.nv, "N": N, "dt": dt,
    "lower": np.asarray(model.lowerPositionLimit).tolist(),
    "upper": np.asarray(model.upperPositionLimit).tolist(),
}, sort_keys=True).encode()).hexdigest()
```

On construction with `build=False`, recompute it from the incoming model and refuse
to load a mismatched `.so` with a message naming what changed. Limits go in the hash
because they are baked into `con_h` bounds — §1.1 says they are subject-independent
today, but the hash is what makes that a checked fact rather than an assumption.

### 3.4 Make `set_model_params` mandatory

`ocp.parameter_values` seeds `p` with the *structural seed subject's* numbers. If a
caller forgets `set_model_params`, the solver runs happily and returns another
person's kinematics. Set `self._params_set = False` in `__init__` and raise from
`solve()` until it is set. Cheap, and it converts the worst failure mode into an
immediate error.

### 3.5 Where the structural seed comes from

Building the parameterized solver still needs *a* model with the marker frames
registered — its numbers become the defaults, only its structure matters. That model
comes from the normal `HumanLoader → scale_human_model → mks_registration` path,
which needs a marker dict.

So ship one: a single reference frame of marker positions committed to the repo
(shipped inside the package), used only to construct the structural model
at image-build time. This removes the last reason the build would need subject data.
It does **not** need to be a real person's calibration — only to contain every marker
name in `SGTS_MKS_MAPPING`, so the frame set is complete.

### 3.6 Runtime flow

```python
# once, at image build (or first startup)
solver = RT_SWIKA_ACADOS_PARAM(structural_model, keys, N, dt, build=True,
                               export_dir=SHIPPED_DIR)

# once per subject, after scale -> register -> IPOPT -> recalibrate
solver.set_model_params(calibrated_model)     # ~0.04 ms, no recompile

# every frame, unchanged
X, U = solver.solve(X, U, marker_meas, X0, cost_weights, dt)
```

`HumanSolver._build_mhe_solver` currently constructs a fresh solver per subject.
For `acados_param` it should instead construct once (lazily, cached on the class or
passed in) and call `set_model_params` on each calibration. That is the only change
outside `ik.py`.

---

## 4. Implementation order

1. **`_build_parameterized(pin_model, keys)`** — returns `(cm, p_sym, p_default,
   joint_ids, frame_ids)`. Pure CasADi, no acados. Testable on its own.
2. **FK regression test** (§5a) against the baked expression. Do this before
   touching acados; it catches every ordering mistake.
3. **`RT_SWIKA_ACADOS_PARAM`** — clone `RT_SWIKA_ACADOS`, swap in the symbolic
   model, add `model.p`, `ocp.parameter_values`, the fingerprint and the
   `_params_set` guard.
4. **`set_model_params(calibrated_model)`** — extract in the §3.1 order, set on all
   stages, flip `_params_set`.
5. **Wire `settings.mhe_backend = "acados_param"`** into `HumanSolver`, building
   once and setting parameters per subject.
6. **Ship it** — build in the Dockerfile to a fixed `export_dir`, construct with
   `build=False` at runtime.

Steps 1–4 are self-contained in `ik.py`; step 5 touches `solver.py` only.

---

## 5. Validation

**5a. FK regression** — parameterized FK at the calibrated `p` must equal the baked
FK. Measured **0.000e+00** over 200 random configurations. This is the guard for
parameter ordering; run it in CI.

**5b. Cross-subject FK** — build from subject A, feed subject B's `p`, compare with
B's own baked FK. Measured **0.000e+00**. This is the actual claim being made.

**5c. End-to-end A/B** — add `acados_param` as a third backend in
`tests/benchmark/benchmark_mhe_ik_backends.py`; per-frame `q` and marker RMSE must
match `acados` to solver tolerance on the same data.

**5d. Two subjects, one `.so`** — calibrate subject A, run; call `set_model_params`
with subject B, run; confirm B's trajectory matches a freshly-baked B solver and
that no compile occurred (check the export dir mtime).

**5e. Fingerprint** — mutate `keys_to_track` and confirm construction with
`build=False` refuses rather than silently mis-mapping.

---

## 6. Residual risks

- **`p_global` untested here.** Stage `p` is validated; do not switch without
  re-running 5a–5d.
- **`keys_to_track` must be fully present.** `build_model` currently does
  `keys = [k for k in KEYS_TO_TRACK if model.existFrame(k)]`. If a subject's model
  were missing a frame, `nmc` would change and the `.so` would be structurally
  invalid. With `acados_param`, assert the full set exists and fail loudly instead
  of silently shrinking the tracked list.
- **Codegen is 57% slower** (20.4 → 32.1 s) and the export dir 11% larger. Both are
  one-time and irrelevant once shipped.
- **fatrop has no C-codegen path any more.** It was removed (dead: reachable only
  through a broken script, wrote `ocp_O3.so` into the working directory, and
  shelled out to gcc with no error checking). `RT_SWIKA_FATROP` is now Python-only
  and rebuilds its CasADi function per subject in ~0.4 s, which is cheap enough to
  leave alone. `settings.ik_code` is gone with it.
- **`dt` and `N` stay baked.** Changing either still requires a rebuild. They are
  config, not subject properties, so this is fine — but the fingerprint covers them
  so a mismatch is caught.

---

## 7. Environment notes

- acados built from source, pinned commit `8e1a6f856` (v0.5.4-20), with
  `-DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_AUTOMATIC`. Needs
  `ACADOS_SOURCE_DIR` (here `/root/workspace/deps/acados`), its `lib/*.so` on the
  loader path, and `t_renderer` at `<acados>/bin/`.
- **CasADi pitfall:** `pip install acados_template` pulls a pip `casadi` wheel that
  shadows the from-source casadi and breaks `import pinocchio.casadi`. Install with
  `--no-deps`. If broken: `pip uninstall -y casadi`.
- Benchmark hygiene: governor `performance`, no `--display`, compare medians;
  acados `sqp_iter` is data-dependent, so cap with `settings.mhe_max_iter` for
  bounded real-time.

### Reference numbers (i7-8850H, N=10, 272 frames)
fatrop(python) ~31 ms / 9.93 mm · acados(default) ~10 ms / 4.29 mm (spikes to
~200 ms on hard frames) · acados(max_iter=3) ~5 ms / ~3.7 mm · agreement mean
`|Δq|` 1.3e-2 rad.

### Known pre-existing bugs (unrelated)
`pipeline.py` online `mhe` steady-state uses bare `settings.ik_type` (should be
`self.settings.ik_type`) and references an undefined `viz_human`. Never hit because
the default `ik_type='sbs'`. `run_pipeline.py` (offline) is fine.

---

## 8. Integration (implemented)

The parameterization is shared by both backends, because both bake the FK and so
both used to regenerate per subject. `RT_SWIKA_FATROP` builds its OCP through
`opti.to_function([...])`, so adding `p` as one more input makes the fatrop
artefact subject-independent in exactly the same way as the acados one.

### 8.1 Layout

```
src/rtcosmik/ik/ocp_model.py     parameterize / extract_params / fingerprint /
                                 manifest I/O / structural model
scripts/python/core/run_ocp_codegen.py   --backend {fatrop,acados,both}, --check
<repo>/ocp/<backend>/                    generated artefacts + ocp_manifest.json
```

`RTCOSMIK_OCP_DIR` overrides the artefact root. It is an absolute path by
design: the previous acados default was `os.getcwd()/acados_codegen`, so *where
the pipeline was launched from* decided whether a compiled solver was found.

### 8.2 Guarding against silent drift

`ocp_manifest.json` records joint names and types, frame names, the tracked
marker list, the parameter id lists, `nq`, `nv`, `N`, `dt`, `with_freeflyer` and
the position limits. Three enforcement points:

1. **On load** — `check_manifest` refuses a mismatched artefact and names what
   changed (`dt: generated 0.025, now 0.026`).
2. **In CI** — `run_ocp_codegen.py --check` exits non-zero when an artefact is
   stale. This is what catches an edit to `settings.py` or `SGTS_MKS_MAPPING`
   that nobody regenerated for.
3. **At generation** — the script prints the fingerprint and what it built from.

A prebuilt solver additionally refuses to `solve()` before `set_model_params`,
because its parameters otherwise hold the *structural seed's* geometry — it would
run happily on the wrong skeleton.

### 8.3 Runtime

`HumanSolver._build_mhe_solver` now tries the pre-generated artefact first and
applies the subject with `set_model_params`; if none matches it generates one for
this subject and warns, which is the old behaviour. So an install with no
generated OCP still works, just slowly.

The fatrop `code='c'` path is restored, with both of its bugs fixed: the library
goes to the artefact directory instead of the working directory, and the compile
is a checked `subprocess.run` instead of a bare `os.system` that printed a timing
and carried on after a failure. Building the CasADi function is also skipped in
`c` mode, where it was seconds of wasted startup.

### 8.4 What generation costs

The fatrop OCP generates a **26 MB `ocp.c`** — the whole horizon unrolled — and
`cc1` peaks above 4 GB compiling it. That is fine on a workstation and a very good
reason never to do it at runtime, which is what this whole change achieves.

### 8.5 Validation (passing)

`tests/benchmark/validate_ocp_params.py` runs the case this exists for: several
COMFI participants, each solved twice — once on a solver generated from their own
model, once on the shared solver with their geometry set as parameters — then
switches between subjects on the same solver to confirm the switch is stateless.

```
loaded the pre-generated acados OCP in 0.13 s, 237 geometry parameters

1012 (h=1.70 m): regenerate 55.4 s vs set params 0.672 ms | max |dq| = 4.15e-15 rad
1118 (h=1.80 m): regenerate 55.5 s vs set params 0.878 ms | max |dq| = 4.66e-15 rad
1508 (h=1.79 m): regenerate 55.4 s vs set params 0.543 ms | max |dq| = 1.89e-15 rad
4279 (h=1.87 m): regenerate 55.1 s vs set params 0.534 ms | max |dq| = 1.72e-15 rad

switching subjects on one solver, twice each: max |dq| = 0.00e+00, stateless
221 s of regeneration replaced by 2.6 ms of parameter setting
```

### 8.6 Two bugs this validation caught

**acados silently regenerated on every load.** `is_code_reuse_possible` calls
`compare_ocp_formulations`, which raises `AttributeError: 'NoneType' object has
no attribute 'shape'` because acados' own JSON round-trip drops `tol`, `qp_tol`
and `qp_solver_tol_*` (each warns "not in dictionary" on read and comes back
None). The exception is swallowed by a bare `except Exception: return False`, so
it rebuilt every time — 55 s per load — while appearing to work.

Fixed by passing `check_reuse_possible=False` when loading. The manifest check in
`__init__` already verifies compatibility against the fields that matter, and
unlike acados' comparison it does not crash. Load time went 55.24 s → **0.13 s**.

**Solver state leaked between subjects.** acados keeps the iterate and duals
across `solve()` calls, which is what makes warm starting work — but after a
subject change the first frames were pulled toward the previous person's
solution. Symptom: the same solver, same subject, same inputs, run twice,
differed by 3.67e-02 rad. `set_model_params` now calls `reset()`, and repeat
passes are bit-identical.

Neither was visible to the fingerprint check, which is why the end-to-end
cross-subject test earns its place: it is the only thing here that would have
caught either.

---

## 9. Solver profiles (measured)

120 frames of real marker data, 43 dof, N=10. Solve time in ms, marker RMSE in
mm (the cost the OCP minimises), jitter is the median frame-to-frame `|dq|`.

| config | median | p95 | max | RMSE | jitter |
|---|---|---|---|---|---|
| acados SQP it=50 tol=1e-4 *(old default)* | 3.66 | 89.5 | 131.7 | **1.92** | 0.0053 |
| acados SQP it=50 tol=1e-6 | 11.73 | **185.0** | 212.7 | 1.03 | 0.0056 |
| acados SQP it=10 tol=1e-6 | 12.31 | 43.2 | 45.8 | 1.03 | 0.0056 |
| **acados SQP_RTI** | 4.85 | **5.9** | **8.7** | **1.02** | **0.0049** |
| acados SQP it=1 | 2.71 | 3.0 | 3.8 | 1.90 | 0.0067 |
| fatrop it=100 tol=1e-6 | 35.70 | 41.0 | 45.7 | 1.07 | 0.0041 |
| fatrop it=10 tol=1e-4 | 26.86 | 28.8 | 46.5 | 1.09 | 0.0042 |

Three findings.

**The old default was the worst of both worlds.** `tol=1e-4` with a 50-iteration
budget gave both a 131 ms tail *and* the worst marker fit (1.92 mm), because the
tolerance stops the solve before the marker term is properly minimised. Raising
the tolerance to 1e-6 fixes the fit but makes the tail worse (212 ms).

**SQP_RTI is not the usual real-time compromise here.** It matches fully
converged SQP on marker RMSE (1.02 vs 1.03) with slightly *less* jitter, at a
twentieth of the tail. This was checked specifically because a lower RMSE than
the converged solve looked wrong: the first measurement compared RTI against an
SQP "gold" left at the default `tol=1e-4`, which was an unfair baseline. At
matched tolerance the two agree, and the jitter metric rules out RTI simply
tracking noise more closely.

**`dq` against the converged solution is ~4.8e-02 rad for RTI** — but SQP at
`tol=1e-4` differs from `tol=1e-6` by 4.2e-02, the same band. With marker RMSE
equal, that difference lives in directions the cost is flat in, not in tracking
quality.

`ocp_model.SOLVER_PROFILES` therefore ships:

    realtime   acados SQP_RTI                       fatrop it=10  tol=1e-4
    accurate   acados SQP it=10 tol=1e-6            fatrop it=100 tol=1e-6

acados `it=10` rather than `it=50` for `accurate`: it reaches the same answer
(`dq` 7.6e-05) for a quarter of the tail, so the extra budget only buys worse
worst cases.

Because `nlp_solver_type`, `qp_solver` and `globalization` shape the generated C,
a profile is a **separate artefact** under `ocp/<backend>/<profile>/`, and the
solver options are part of the fingerprint -- switching profiles without
regenerating is refused rather than silently reusing the wrong `.so`.
