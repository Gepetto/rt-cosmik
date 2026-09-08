# Paper comparison: mmpose/LSTM vs NLF

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

## Two measurements still owed

**Throughput is not comparable as it stands.** The mmpose arm starts from
COMFI's precomputed 2D keypoints, so its figure covers triangulation, the LSTM
and the IK, and excludes the cost of running mmpose itself. NLF's figure is
end to end: YOLO, NLF, video decode and IK. Reporting 90 fps against 32 fps
would therefore be wrong in the baseline's favour. Either label the baseline's
number as post-2D, or measure mmpose and add it.

mmpose is not installed here, and `tests/benchmark/benchmark_mmpose_batched_all.py`
-- which already targets these six tasks -- needs the detector and pose weights.
Its results were never committed, so someone may already have them.

**The NLF timings in this sweep are lower bounds.** They were measured with
another job sharing the GPU: about 2.9 GB of the 5.07 GB in use was outside this
container. Accuracy is unaffected, timing is not. Re-benchmark on a free GPU
before quoting any throughput number.
