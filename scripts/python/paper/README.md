# Paper comparison: mmpose/LSTM vs NLF vs FastSAM

Everything here exists to produce one number per architecture on equal terms.
None of it is part of the live toolbox.

## The parity idea

The two front ends do not emit the same markers. 29 of the LSTM augmenter's 43
are the same anatomical landmarks NLF produces; adding the 6 face keypoints both
emit natively gives **35 markers the baseline can actually produce**. The 8 that
only NLF has -- `T11`, `T6` and the six hand markers -- leave 7 DoF unobservable:

| lost markers | DoF | why |
|---|---|---|
| the six hand markers | `left/right_wrist_Z`, `left/right_wrist_X` | nothing distal to the wrist |
| `T6` | `middle_thoracic_Z/X/Y` | C7 alone is one point; orientation survives only indirectly, through the shoulders past a clavicle DoF each |
| `T11` | none | the pelvis keeps four markers |

So `marker_set = "parity"` in `settings.py` gives both arms the same 35 markers,
the same model with those 7 DoF locked, and the same IK. **36 articulated DoF,
29 scored.** `marker_set = "nlf"` restores the ordinary 43-marker, nothing-locked
configuration.

Locking goes through the joint position limits, not a reduced model: both
backends already enforce them, the limits are fingerprinted so a mismatched OCP
is refused rather than silently reused, and `nq`/`nv` stay put so frame ids and
CSV columns still line up with an unlocked run. The window is `1e-4` rad rather
than zero, because a zero-width box has no interior for acados' interior-point
QP -- see `apply_joint_locks`.

Artefact paths carry a non-default marker set (`ocp/acados/realtime_parity`), so
the parity and ordinary OCPs coexist.

## The FastSAM arm

COMFI also ships a FastSAM-based metric 3D export, one CSV per trial under
`fastsam/<participant>/<task>/cosmik_mhr_markers_cam.csv`, **already in the
reference camera's frame** and for camera 0 only. So this arm runs no network:
`src/rtcosmik/paper/fastsam_source.py` reads the file, applies the same
`p_world = R p_cam + T` anchor every arm applies, low-passes with the same IIR,
and hands the markers to the same solver.

Its marker set is a near-exact parity match, which is what makes the comparison
fair without tuning. 34 of the 35 parity markers are present under identical
names; `TV8` and `TV12` are extra and dropped, since parity already drops `T11`
and `T6` and locks the thoracic DoF. The one genuine gap is `Head`, which cannot
just be omitted -- `construct_segments_frames` only builds the head segment when
Head, REar and LEar are all present, so dropping it would silently give this arm
a structurally different model.

`Head` is therefore reconstructed from the facial landmarks FastSAM does export,
at an offset **measured** from the NLF runs rather than guessed, and taken from
NLF rather than from mocap so that nothing is tuned toward the reference. It is
not a load-bearing choice: deliberately wrong placements move the whole-body
RMSE by at most 0.18 deg, against a ~2 deg spread between arms.

```bash
python3 scripts/python/paper/study_head_offset.py measure       # where Head sits
python3 scripts/python/paper/study_head_offset.py sensitivity   # does it matter
bash    scripts/bash/run_fastsam_sweep.sh /root/workspace/COMFI
```

**Its fps column is not comparable with the others.** FastSAM's own inference is
not run here -- the 3D arrives precomputed -- so the figure covers the IK and
nothing else. It is an IK throughput number, not a pipeline one.

## The SMPL-refinement arm (`nlfsmpl`)

NLF regresses each canonical SMPL-X vertex independently, so nothing forces its
output to be a body a human could have. `docs/smplfitter.md` covers fitting the
SMPL model back to those points; this arm puts that between NLF and the IK.

The correspondence is free -- NLF's canonical vertices *are* SMPL-X vertices,
index for index -- so the arm asks NLF for all 10475 instead of the 35 markers,
fuses the views as usual, fits the body, and reads the same
`settings.nlf_indices` rows off the **fitted** vertices. Only the fit differs
from the `nlf` arm.

Measured cost of the dense output on an RTX 4500 Ada, before any fitting:

| cameras | 35 markers | 10475 vertices |
|---|---|---|
| 1 | 9.40 ms | 10.15 ms |
| 4 | 18.60 ms | 22.95 ms |

**Fitting only the vertices we need does not work**, though it is the obvious
saving. Against an exact synthetic target the full model converges to 0.65 mm;
a 1024-vertex decimated subset plus the 35 markers reaches 148.9 mm at 4
iterations, 81.3 at 8 and 65.6 at 16, and smplfitter's own `vertex_subset_size`
path is no better (109.3 mm at 512). The distributed models omit the
`vertex_subset_joint_regr_post_lbs_N.npy` that path expects, so the joint
regressor gets sliced column-wise and the LBS joints -- which drive the fit --
land in the wrong place. Renormalising the slice makes it worse. Requires
`pip install trimesh fast_simplification` to reproduce.

Three shape policies, via `--beta-mode`: `free` refits the shape every frame,
`calibrated` fits it once over the first `smpl_calibration_frames` and then holds
it with `fit_with_known_shape` (causal, and how a real session would run), and
`shared` fits one shape over the whole trial offline as an upper bound that
cannot ship.

```bash
bash    scripts/bash/setup_smplfitter.sh          # one-off, needs registration
python3 scripts/python/paper/check_smplfitter.py  # verifies every silent trap
python3 scripts/python/paper/study_smpl_fit.py    # baseline vs the shape policies
```

**Body models are licence-gated.** smplfitter needs SMPL-X files that require a
free registration at `smpl-x.is.tue.mpg.de` (plus smpl, mano and agora, same
email and password); the downloader authenticates as you. There is no
redistributable copy, so this arm cannot run in a fresh container until
`SMPLFITTER_BODY_MODELS` points at a downloaded copy.

This container runs torch 2.4.1, and smplfitter 0.5 needs `torch.nn.Buffer` from
torch 2.5. `rtcosmik.smpl.torch_shim` supplies it and must be imported first;
`rtcosmik.smpl.fitter` does that for you.

## Running it

```bash
# settings.py: marker_set = "parity"
python3 scripts/python/core/run_ocp_codegen.py --backend acados     # once per N

# one trial, verbose
python3 scripts/python/paper/run_mmpose_baseline.py \
    --dataset /root/workspace/COMFI --participant 1012 --task Lifting

# a whole arm, resumable, one summary row per trial
python3 scripts/python/paper/sweep.py --arm mmpose --cameras 0 2 4 6 \
    --dataset /root/workspace/COMFI --summary results/mmpose_4cam.csv
python3 scripts/python/paper/sweep.py --arm nlf --cameras 0 2 4 6 \
    --dataset /root/workspace/COMFI --summary results/nlf_4cam.csv

# the comparison tables
python3 scripts/python/paper/report.py results/*.csv --by-task
```

`sweep.py` skips trials already in the summary, so it can be stopped and
restarted, and one failing trial does not lose the batch.

## Fairness, deliberately

- **Causal.** The LSTM sees a 30-frame window of *past* frames only, as the old
  real-time pipeline used it. Running it over a whole trial at once -- the usual
  offline OpenCap usage -- would let the baseline see the future while NLF works
  frame by frame.
- **Weighted.** mmpose confidence enters the DLT as `1/score`, so the baseline
  gets the same uncertainty-weighted multi-view fusion NLF gets.
- **Same filter.** The codebase's own IIR, with `settings` values, applied by the
  same buffer procedure in both arms.
- **Same everything downstream.** Model, calibration, IK, evaluation.

One asymmetry is real and worth stating rather than hiding: the mmpose arm needs
at least two views to triangulate, so the 1-camera row exists only for NLF.

## Known task effect

`StraightWalking` scores far worse than other tasks (~24 deg against ~13-16 deg)
for both subjects checked. It is not a time-alignment artefact -- forcing lag 0
changes the number by under a degree. Cameras 4 and 6 see the subject poorly in
that trial (mean confidence 0.59/0.64 against 0.83/0.85 for cameras 0 and 2), so
effective coverage drops to ~3.7 of 4 cameras. It is unfavourable rig geometry
for that walk direction, and it applies to both arms equally.

## Reading the timing columns

**The sweep's mmpose fps column is post-2D only.** That arm reads COMFI's
precomputed 2D keypoints, so its figure covers triangulation, the LSTM and the
IK, and excludes running mmpose. NLF's figure is end to end: YOLO, NLF, video
decode and IK. The two must not be printed side by side without adding mmpose's
own inference cost, which is measured separately and held outside this repo.

**The NLF timings in this sweep are lower bounds.** They were measured with
another job sharing the GPU: about 2.9 GB of the 5.07 GB in use was outside this
container. Accuracy is unaffected, timing is not. Re-benchmark on a free GPU
before quoting any throughput number.
