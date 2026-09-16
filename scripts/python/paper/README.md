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

COMFI also ships a FastSAM-based metric 3D export, one CSV per trial and camera
under `fastsam/<participant>/<task>/cosmik_mhr_markers_cam{0,2,4,6}.csv`, each
**in its own camera's frame** (cameras 2/4/6 are exported from
`fastsam/results_multicam` by `export_fastsam_multicam.py`). So this arm runs no
network: `src/rtcosmik/paper/fastsam_source.py` reads the files, fuses the views
in the reference camera's frame the way NLF fuses its per-view 3D (a plain mean,
since FastSAM gives no per-point uncertainty), applies the same
`p_world = R p_cam + T` anchor every arm applies, low-passes with the same IIR,
and hands the markers to the same solver. It runs with 1, 2 and 4 cameras.
Participant 3361 is excluded: its FastSAM export does not match COMFI's
calibration.

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
```

**Its fps column is not comparable with the others.** FastSAM's own inference is
not run here -- the 3D arrives precomputed -- so the figure covers fusion and IK
and nothing else. Its inference time is read from the export logs instead
(`fastsam_timing.py`): about 390 ms per view.

## NLF marker vertices

NLF is queried at SMPL-X canonical vertices, one per marker. They were first
picked by hand; `settings.nlf_indices` now holds vertices fitted to mocap with
the method that produced FastSAM's marker map (`fit_nlf_marker_map.py`: dense
NLF samples, per-frame similarity alignment to the Vicon markers, balanced
median over frames, tasks and participants, then a one-to-one assignment; fit on
14 participants, checked on 3 held out, refit on all 17). Hands and face are not
fitted, and the pelvis keeps its hand-picked vertices: the fitted posterior
markers tilt the pelvis frame about 8 deg against mocap, and with the thoracic
joints locked the thorax follows, which cost up to 3 deg of whole-body RMSE and
added shoulder flips in overhead work (details in the script's docstring). The
hand-picked values are kept as comments in `settings.py`.

## SMPL refinement: tested and rejected

Fitting a SMPL body to NLF's dense vertex output, between NLF and the IK, was
built and measured, then removed. Keeping the numbers so nobody repeats it; the
implementation is in git history if it is ever wanted.

Joint RMSE against the mocap reference, 6 trials:

| configuration | 1 camera | 4 cameras |
|---|---|---|
| NLF, no fit | **11.76** | **10.92** |
| fit fused cloud, beta per frame | 11.79 | 11.28 |
| fit fused cloud, beta calibrated | 11.60 | 11.17 |
| fit each view then fuse, calibrated | — | 11.24 |

It helps only at one camera, by 0.16 deg, for 13.1 ms a frame -- 73 to 37 fps.
Adding three cameras instead costs 9.0 ms and gains 0.84 deg, roughly five times
the accuracy per millisecond. At four cameras every variant is worse than not
fitting.

The mechanism: the fit cuts segment-length wander 25-34% (10.4 -> 7.8 mm at four
cameras, against mocap's own 7.1) but moves segment-length *bias* by under 2 mm.
Wander is what fusion and the IK already handle -- the solver fixes segment
lengths at calibration -- while bias is what actually costs joint accuracy. So it
re-imposes a constraint that is already satisfied and leaves the error that is
not. NLF's output already sits within 6-8.6 mm of a valid SMPL-X body, which is
why there was so little for the fit to change.

That also rules out swapping in a more expressive body model: expressiveness is
not the limit. The lever is a better regressor -- see the FastSAM arm, the only
modality with both low wander (6.4 mm) and low bias (6.3 mm).

`docs/smplfitter.md` is kept: it is a standalone guide, useful independently.

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

# the whole campaign: every arm, one after another, then every metric
bash scripts/bash/run_campaign.sh validation 1012     # one participant first
bash scripts/bash/run_campaign.sh campaign            # everyone
```

`run_campaign.sh` writes runs to `output/<campaign>/` and everything else to
`results/<campaign>/`: the sweep summaries, `vs_mocap/` (rescored against the
mocap reference), and `paper/`, which holds per-trial tables (`trial_metrics.py`,
`reba_agreement.py`, `robot_distance.py`, `fastsam_timing.py`) and the aggregated
tables and statistics (`aggregate_results.py`: participant means, mean (SD),
Friedman then Wilcoxon with Holm correction).

`sweep.py` skips trials already in the summary, so it can be stopped and
restarted, and one failing trial does not lose the batch.

## Fairness, deliberately

- **Causal.** The LSTM sees a 30-frame window of *past* frames only, as the old
  real-time pipeline used it. Running it over a whole trial at once -- the usual
  offline OpenCap usage -- would let the baseline see the future while NLF works
  frame by frame.
- **Weighted.** mmpose confidence enters the DLT as `1/score`, so the baseline
  gets the same uncertainty-weighted multi-view fusion NLF gets.
- **Same filter.** The codebase's own IIR, with `settings` values, fed one frame
  at a time (`MarkerFilter`) in every arm.
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

## REBA and human-robot distance

**Posture REBA** (`reba_agreement.py`, `rtcosmik/ergonomics/reba_posture.py`) is
computed per frame for the arm and for the reference and compared: score error,
risk-level agreement and weighted kappa, time-in-level error, per-component
agreement. Angles are taken relative to a neutral posture: every COMFI trial
starts with the participant in the calibration pose, so the neutral is the
median over each trial's first 0.5 s -- either the reference's (offsets of the
arm count as error) or the arm's own (what a deployed system would do).

**Human-robot distance** (`robot_distance.py`) uses COMFI's Panda joint states,
which carry camera 0's timestamps and so match video frames exactly. In both
robot tasks the participant hand-guides the robot, so a whole-body minimum
distance is mostly zero; the script also reports the distance of the body
without forearms and hands (usually the head) and each hand's distance to the
robot's hand frame. The human is the model's joint-centre skeleton, rebuilt per
run exactly as the IK scaled it.

## Reading the timing columns

**The sweep's mmpose fps column is post-2D only.** That arm reads COMFI's
precomputed 2D keypoints, so its figure covers triangulation, the LSTM and the
IK, and excludes running RTMPose. `aggregate_results.py` adds RTMPose's own
cost from Table I of the RT-COSMIK draft (7.1 ms for 2 cameras, 13.2 ms for 4,
same machine) before quoting a rate. NLF's figure is end to end: YOLO, NLF,
video decode and IK. FastSAM's rate is its logged inference time per view times
the number of views, plus the IK.

These are offline throughputs over the dataset, good enough to separate what
runs at 30 Hz or more from what can only run offline; they are not a latency
benchmark of the live pipeline. Arms run one at a time on a GPU nothing else is
using -- an earlier sweep shared the GPU with another job and its NLF timings
were lower bounds.
