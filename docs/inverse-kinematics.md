# Inverse kinematics

RT-COSMIK fits its human model to the fused landmarks with one of two inverse
kinematics methods, chosen by `ik_type` in `settings.py`:

| `ik_type` | Method | When to use it |
|---|---|---|
| `"mhe"` (default) | Moving-horizon estimation: the model is fitted to the landmarks of the last `N` frames at once, under joint limits, with a smoothness cost. The newest pose is returned. | Always, unless you need the last millisecond. On COMFI it was 1.4° more accurate than `"sbs"` for 1.7 ms more per frame. |
| `"sbs"` | Sample by sample: a damped quadratic program per frame. | Debugging, or a machine without a generated solver. |

The moving-horizon problem is

```math
\begin{aligned}
\min_{\mathbf{x}_j,\,\ddot{\mathbf{q}}_j}\;& \sum_{j=0}^{N-1} w_y\sum_{i}\big\|\mathbf{y}_{j,i}-\boldsymbol{\phi}_i(\mathbf{q}_j,\boldsymbol{\theta})\big\|^2
+ \sum_{j=0}^{N-1} w_x\big\|\mathbf{x}_j \ominus \bar{\mathbf{x}}\big\|^2 + \sum_{j=0}^{N-2} w_u\big\|\ddot{\mathbf{q}}_j\big\|^2\\
\text{s.t.}\;& \mathbf{q}_{j+1} = \mathbf{q}_j \oplus \dot{\mathbf{q}}_j\,dt,\qquad
\dot{\mathbf{q}}_{j+1} = \dot{\mathbf{q}}_j + \ddot{\mathbf{q}}_j\,dt,\qquad
\boldsymbol{\alpha}^- \le \boldsymbol{\alpha}_j \le \boldsymbol{\alpha}^+
\end{aligned}
```

where $\mathbf{y}_{j,i}$ is landmark $i$ at frame $j$, $\boldsymbol{\phi}_i$ the
position of the same landmark on the model, $\boldsymbol{\theta}$ the person's
anthropometry, $\bar{\mathbf{x}}$ the previous estimate and $\boldsymbol{\alpha}$
the joint angles. The weights $(w_y, w_x, w_u)$ are `cost_weights` in
`settings.py`, and the horizon is `N`.

## Backends

`mhe_backend` picks the solver of the moving-horizon problem:

- `"acados"` (default): one real-time iteration per frame (`SQP_RTI`), warm-started
  from the previous solution.
- `"fatrop"`: an interior-point solver, the validated reference.

## Generating the solver

Only for `ik_type = "mhe"`. Like the detector engines, these are compiled
artefacts: generated once, gitignored, never committed.

```bash
python3 scripts/python/core/run_ocp_codegen.py                 # everything
python3 scripts/python/core/run_ocp_codegen.py --backend acados --profile realtime
python3 scripts/python/core/run_ocp_codegen.py --check         # up to date?
```

Output goes to `ocp/<backend>/<profile>/`. Expect ~55 s per acados artefact and
~4 min per fatrop one, which compiles a 26 MB C file.

The optimal control problem is parameterized by the subject's geometry (segment
lengths and marker offsets), so **one generated solver serves every person**:
calibrating a new subject sets parameters in under a millisecond instead of
recompiling for 20-40 s mid-session. Generation needs no subject data at all,
only the model's topology, which is the same for everybody.

Each artefact carries an `ocp_manifest.json` recording what it was built from.
The pipeline refuses to load one that no longer matches your configuration, and
names what changed. Re-run after changing the tracked marker set, `N`, the model,
or the marker-to-joint mapping, plus `fs` for acados, which bakes the timestep
into its dynamics (fatrop takes it at runtime). `--check` answers this and is
what CI should call.

## Speed/accuracy profile

`settings.mhe_profile` picks the solver configuration:

| | per-frame solve (median / p95 / max) | marker RMSE |
|---|---|---|
| `realtime` (acados `SQP_RTI`) | 4.9 / 5.9 / 8.7 ms | 1.02 mm |
| `accurate` (acados `SQP`, 10 iterations, tol 1e-6) | 12.3 / 43.2 / 45.8 ms | 1.03 mm |

Measured over 120 frames of real data, 43 dof, N=10. `realtime` bounds the
per-frame cost: plain `SQP` has the same median but a 130-210 ms tail on hard
frames, which breaks a 40 fps budget. The two agree on marker fit to 0.01 mm, and
`realtime` is marginally *smoother* frame to frame, so it is the default.

The profile is baked into the generated code, so switching it needs a
regeneration; `--check` will say so.
