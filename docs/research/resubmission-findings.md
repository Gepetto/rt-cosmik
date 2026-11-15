# Resubmission findings to discuss before rewriting the paper (ticket 21)

State on 2026-11-13, with updates on 2026-11-14 (section 0) and 2026-11-15 (sections 0b and 0d; 0d holds the current figures). Every number is on COMFI, 18 participants, six tasks, 108 trials, four cameras (0, 2, 4, 6), out-of-fold landmark maps, participant mean ± SD unless stated. "Causal" error compares estimate and reference at the same frame after removing only the per-trial clock offset between mocap and video (2 or 3 frames). "Aligned" error first removes the lag estimated on knee flexion, as the submitted paper did.

Where the data are: runs in `/root/workspace/oof/runs/<id>/<task>/<arm>/`, tables in `results/oof/` of the main checkout and in `/root/workspace/oof/t11`, `t12`, `t13` (not in git).

## 0. Update of 2026-11-14: shoulder branch, mean fusion, FastSAM-3D on matched cameras

This section supersedes the figures of sections 1, 3, 5 and 7 where they differ. The tables of sections 2 and 4 are superseded by the replays at the end of this section; the speed figures of section 4 stand. Runs of this section are in `/root/workspace/oof/runs26`, tables in `/root/workspace/oof/t26`, `t12/v3` and `t13/v3`.

**Shoulder branch (ticket 26).** The shoulder is three angles in the order flexion, abduction, rotation, so a raised arm has two descriptions: `(z, x, y)` and `(z + 180, 180 − x, y + 180)`. With one solver iteration per frame, the IK stays in the description it entered near 90° of abduction; a flexion or rotation limit then holds it and the arm stops following the markers. It happened in 12 of the 108 trials in at least one arm, all in SideOverhead (12 of its 18 trials), and in the reference too (7 arm-trials). Two changes:

- The IK moves a shoulder to the other description when it is within 3° of a limit and the other description fits the limits. Trials without a flip are unchanged (median difference 0.000°).
- Joint error is measured with the estimate's shoulder in the description nearer the reference's at each frame.
- Tested and rejected: wider flexion and rotation limits (a trial without flips went from 9.1° to 10.4°), and a free flexion limit alone (10.9° against 10.1° for the adopted rule on eight trials; the rotation limit still holds the shoulder). A ball joint at the shoulder would remove the cause; it changes the model, the solver, the saved angles and every script that names the shoulder joints, and is left for after the resubmission.

**Results with this solver, causal, 18 participants.**

| Arm | Cameras | Causal | Aligned | SideOverhead | Marker error |
|---|---|---|---|---|---|
| NLF-S, unweighted mean | 0, 2, 4, 6 | 8.70° ± 0.84° | 8.45° | 9.79° | 54 mm |
| NLF-S, inverse variance | 0, 2, 4, 6 | 8.81° ± 0.87° | 8.56° | 9.91° | 53 mm |
| NLF-S, unweighted mean | 2, 4, 6 | 8.89° ± 0.83° | 8.66° | 10.01° | 65 mm |
| NLF-S, unweighted mean | 2, 4 | 8.95° ± 0.85° | 8.68° | 10.08° | 57 mm |
| NLF-S, unweighted mean | 4, 6 | 9.53° ± 0.89° | 9.32° | 10.68° | 112 mm |
| NLF-S, unweighted mean | 2 | 9.57° ± 0.91° | 9.35° | 11.16° | 109 mm |
| FastSAM-3D | 2, 4, 6 | 8.50° ± 1.08° | 8.22° | 10.68° | 55 mm |
| FastSAM-3D | 2, 4 | 8.52° ± 1.07° | 8.21° | 10.69° | 47 mm |
| FastSAM-3D | 4, 6 | 9.00° ± 1.08° | 8.80° | 11.02° | 94 mm |
| FastSAM-3D | 2 | 9.25° ± 1.10° | 8.93° | 11.89° | 91 mm |

- Flip trials (more than 1 % of frames): 0 in every arm, except 1 for NLF-S on camera 2.
- **Fusion.** Mean against inverse variance: −0.11°, signed-rank p = 0.97. The exponent of the weighting chosen on each fold's 15 fitting participants is 0 (the mean) in 5 of 6 folds; on the held-out participants it gives 8.91° against 9.14° (earlier solver). Weighting does not help the base position (41.2 mm both) or orientation (6.66° against 6.50°, not significant). Per-marker mixes of the two rules are not better. The unweighted mean is now the default (`settings.fusion`).
- **NLF-S against FastSAM-3D on the same cameras and the same fusion** (2 / 4-6 / 2-4 / 2-4-6): FastSAM-3D is lower by 0.32° / 0.53° / 0.43° / 0.40° (signed-rank p = 0.20 / 0.07 / 0.07 / 0.13 on participant means; 0.5° to 0.7° and p < 0.01 in three of four on the median over trials). By task on 2-4-6, FastSAM-3D is better on Screwing (7.64° against 9.48°) and Polishing (8.49° against 9.30°), NLF-S on SideOverhead. Supported wording: NLF-S is chosen for speed; FastSAM-3D, a slower foundation model, is about 0.4° more accurate on the same cameras; the two are equivalent within ±1°.
- **Alternatives to fusion followed by IK**, on six validation participants: an IK fitting each view's 2D keypoints directly, 11.19° against 9.01° (NLF-S's 2D keypoints disagree across views by 5 to 8 pixels); an IK fitting each view's 3D pose directly, 9.01°, identical to mean fusion followed by IK and not faster. Both dropped.
- **Tiled detection**, 108 trials, earlier solver: 107 trials within 0.22° of per-view detection; one trial (4665 RobotWelding) went from 8.8° to 21.7° through a shoulder flip, which the new solver removes on that trial (8.7°). Confirmed by the replay below.
- **REBA** (section 5) now reads shoulder flexion and abduction in the description with abduction within 90°, and scores an arm above the horizontal as above shoulder level. Frames with abduction within 90° keep their score; 0.2 to 0.4 % of frames change, all in SideOverhead. Regenerated, causal, mean fusion: mean absolute error 1.07 / 0.91 / 0.72 (absolute / reference's neutral / own neutral), same risk level 59 % / 69 % / 75 %, weighted kappa 0.46 / 0.61 / 0.70; inverse variance: 1.01 / 0.92 for the first two, same risk level 61 % / 68 %. High-risk frames rated lower: 46 % (mean, absolute) and 38 % (inverse variance, absolute). The conclusions of section 5 stand.
- **Hand-to-robot distance** (section 6) regenerated: unchanged to 0.1 mm in every pooled figure (the robot trials have no flip in these arms).

**Filter and IK studies, replayed with this solver and the mean** (supersedes the table of section 2; 108 trials per variant, 0 solver failures, 0 dropped frames):

| Variant | Causal | Aligned | Lag (frames) | Jitter |
|---|---|---|---|---|
| MHE, horizon 7, 5 Hz order 4 (paper setting) | 8.70° ± 0.84° | 8.45° | 2.9 | 0.156° |
| Per-frame IK, same filter | 10.31° ± 1.59° | 10.04° | 3.3 | 0.154° |
| MHE, horizon 3 / 5 / 10 / 15 / 20 | 8.68 / 8.69 / 8.71 / 8.73 / 8.74° | 8.45 to 8.46° | 2.6 to 3.0 | 0.165 to 0.147° |
| MHE, no filter | 8.45° | 8.46° | −0.4 | 0.279° |
| MHE, 3 / 4 / 8 / 10 Hz, order 4 | 9.07 / 8.83 / 8.54 / 8.50° | 8.45 to 8.46° | 5.4 / 3.8 / 1.5 / 0.9 | 0.133 / 0.145 / 0.186 / 0.203° |
| MHE, order 2 at 6 / 8 / 10 Hz | 8.50 / 8.48 / 8.47° | 8.46° | 1.0 / 0.5 / 0.4 | 0.169 / 0.187 / 0.204° |

- MHE against per-frame IK: +1.61° [1.13, 2.09], Holm p < 0.001, not equivalent. The per-frame IK has the shoulder rule too (10.31° ± 1.59°; 10.34° without it), so the gap is not an effect of that rule. It was +1.13° before ticket 26.
- Horizon length: every horizon from 3 to 20 is within 0.04° of horizon 7.
- The filter still does not change aligned error; it trades lag against jitter. No filter: −0.25° causal, jitter 0.279° against 0.156°. Order 2 at 6 Hz: −0.20° causal, lag 1.0 frame, jitter 0.169°.

**TensorRT and tiled detection, replayed with this solver and the mean** (supersedes the accuracy bullets of section 4; 108 trials each, no flip trial):

| Inference | Causal | Aligned | Largest difference to TorchScript on one trial |
|---|---|---|---|
| TorchScript, per-view detection | 8.70° ± 0.84° | 8.45° | |
| TensorRT, per-view detection | 8.70° ± 0.84° | 8.45° | 0.05° |
| TensorRT, tiled detection | 8.70° ± 0.83° | 8.45° | 0.21° |

The three are interchangeable for accuracy. The trial that tiled detection degraded with the earlier solver (4665 RobotWelding) is now within 0.21° like the others.

**Arms by replay of the recorded views (ticket 08)**, 108 trials each, causal:

| Arm | Cameras | Causal | Aligned | Marker error | Jitter |
|---|---|---|---|---|---|
| NLF-S, 3D fused | 0, 2, 4, 6 | 8.70° ± 0.84° | 8.45° | 55 mm | 0.16° |
| NLF-S, 3D fused | 0, 4 (opposed) | 8.91° ± 0.88° | 8.66° | 62 mm | 0.19° |
| NLF-S, 3D fused | 0, 2 (same side) | 9.21° ± 0.92° | 8.97° | 108 mm | 0.19° |
| NLF-S, 3D fused | 0 | 9.50° ± 0.98° | 9.29° | 118 mm | 0.24° |
| NLF-S, 2D triangulated | 0, 2, 4, 6 | 9.57° ± 0.97° | 9.32° | 74 mm | 0.22° |
| NLF-S, 2D triangulated | 0, 2 | 12.96° ± 1.25° | 12.76° | 96 mm | 0.47° |
| NLF-S, 2D triangulated | 0, 4 | 22.18° ± 8.19° | 22.08° | 105 mm | 0.29° |
| RTMPose+LSTM (the baseline of the submitted paper) | 0, 2, 4, 6 | 12.81° ± 1.07° | 12.53° | 80 mm | 0.20° |

The arms on cameras 2 / 4-6 / 2-4 / 2-4-6 of the first table are now produced by the same repository script, with the same figures.

**MHE weights** (state and control weight a decade either side of 1e-3 and 1e-5): control weight, no effect (8.70° both ways); state weight 1e-4, 9.68° ± 1.52°; state weight 1e-2, 12.25° ± 1.05°. The result depends on the state weight within a decade; the text must not claim robustness to it.

**Fusion rules, all four on this solver:** mean 8.70°, inverse variance 8.81°, reprojection weights 9.18°, reprojection weights with symmetry refinement 9.83°.

**Statistics (ticket 11), on participant means, Holm within each family, equivalence at ±1°:**

- Arms (10 comparisons). Proposed against mmpose: −4.12° [−4.68, −3.55], p < 0.001. 3D fusion against 2D triangulation, four cameras: −0.87° [−1.28, −0.46], p = 0.001. Camera ladder: 1 against 2 same-side +0.29°, 2 same-side against 4 +0.51°, 1 against 4 +0.80°, same-side against opposed pair +0.30°, all p ≤ 0.014. FastSAM-3D against NLF-S on matched cameras: −0.32°, −0.53°, −0.43°, −0.40°, none significant (Holm p = 0.27 each); equivalence within ±1° is not established after correction either (Holm p = 0.07 to 0.11).
- IK and filter (14 comparisons): per-frame IK as above; every horizon equivalent to 7 (differences up to 0.04°, significant); every filter variant equivalent to the paper setting and significantly different (−0.25° without filter, +0.37° at 3 Hz).
- MHE weights (4): state weight 1e-4 +0.98° [0.54, 1.42], p = 0.002; 1e-2 +3.55°, p < 0.001; control weight equivalent, not different.
- Fusion against the mean (3): inverse variance +0.11°, p = 0.97, equivalent; reprojection weights +0.48° [0.12, 0.84], p = 0.028, equivalent; with symmetry refinement +1.13° [0.67, 1.58], p < 0.001.
- The families were redeclared on this date: fusion rules are compared with the mean (the headline rule) and FastSAM-3D with NLF-S on the cameras that have meshes.

**1-euro filter (ticket 09).** Grid of 20 points (cutoff at rest 0.5 to 5 Hz, beta 0.5 to 15), 108 trials each, no solver failure. The point chosen on each fold's fifteen fitting participants (lowest causal error with jitter not above the 5 Hz Butterworth's) is the same in all six folds: 2 Hz at rest, beta 15.

| Filter | Causal | Aligned | Lag (frames) | Jitter |
|---|---|---|---|---|
| 5 Hz Butterworth, order 4 (submitted) | 8.70° | 8.45° | 2.9 | 0.156° |
| None | 8.45° | 8.46° | −0.4 | 0.279° |
| 1-euro, cross-validated | 8.46° | 8.46° | 0.4 | 0.153° |

By the rule recorded before these runs (no filter unless the 1-euro arm has lower causal error), the headline is the MHE without filter, by 0.007°. The 1-euro arm reaches the no-filter accuracy with the Butterworth's jitter. **Decided (2026-11-14):** the headline results use the tuned 1-euro filter. The paper reports the three (none, 5 Hz Butterworth, tuned 1-euro) and says the 1-euro filter gives less jitter than the Butterworth without its lag and the accuracy the lag costs. Every arm is to be regenerated with this filter.

**Fusion validation (ticket 10)**, details in `docs/research/fusion-validation.md`:

- NLF's uncertainty is weakly related to a view's 3D error: rank correlation 0.13 over time and across views; the lowest-uncertainty view is the best one 29 % of the time (chance 25 %). Same on robot tasks.
- Held-out participants (12), causal error, mean / inverse variance / median: clean 8.60 / 8.66 / 8.69°; views masked half the time 9.21 / 9.27 / 9.39°; a 1 m outlier on one limb of one view 11.64 / 11.63 / 8.90°.
- The unweighted median is offered as an option (`settings.fusion = "median"`): about 0.1° worse on clean data, and it removes a gross outlier (−2.75°, p < 0.001). It gives nothing at 0.3 m or below. The uncertainty-weighted median is unusable (15 to 18°).
- Limits: the outliers are synthetic; how often NLF produces such errors on real data was not measured; these p-values are unadjusted.

## 0b. Update of 2026-11-15: every result with the tuned 1-euro filter

The paper setting is now the 1-euro filter at 2 Hz at rest, beta 15 (commit `209511c`). Everything below was regenerated from that commit: 108 trials per arm, no solver failure, no dropped frame. These figures supersede section 0 wherever a figure there was produced with the 5 Hz Butterworth; the reference is unchanged. Runs in `/root/workspace/oof/runs27` (arms) and `runs26` (study variants), tables in `/root/workspace/oof/t27`.

**Arms, causal, 18 participants.**

| Arm | Cameras | Causal | Aligned | Marker error | Jitter |
|---|---|---|---|---|---|
| NLF-S, 3D fused (headline) | 0, 2, 4, 6 | 8.46° ± 0.84° | 8.46° | 52 mm | 0.15° |
| NLF-S, 3D fused | 0, 4 (opposed) | 8.67° ± 0.88° | 8.66° | 59 mm | 0.19° |
| NLF-S, 3D fused | 0, 2 (same side) | 8.98° ± 0.92° | 8.97° | 105 mm | 0.19° |
| NLF-S, 3D fused | 0 | 9.30° ± 0.98° | 9.29° | 116 mm | 0.24° |
| NLF-S, 2D triangulated | 0, 2, 4, 6 | 9.33° ± 0.99° | 9.32° | 70 mm | 0.22° |
| NLF-S, 2D triangulated | 0, 2 | 12.84° ± 1.28° | 12.83° | 94 mm | 0.49° |
| NLF-S, 2D triangulated | 0, 4 | 22.41° ± 8.34° | 22.43° | 102 mm | 0.28° |
| RTMPose+LSTM (the baseline of the submitted paper) | 0, 2, 4, 6 | 12.57° ± 1.08° | 12.53° | 77 mm | 0.19° |
| NLF-S, 3D fused | 2, 4, 6 | 8.66° ± 0.84° | 8.66° | 61 mm | 0.17° |
| NLF-S, 3D fused | 2, 4 | 8.70° ± 0.85° | 8.69° | 54 mm | 0.19° |
| NLF-S, 3D fused | 4, 6 | 9.33° ± 0.89° | 9.32° | 106 mm | 0.19° |
| NLF-S, 3D fused | 2 | 9.35° ± 0.91° | 9.35° | 102 mm | 0.24° |
| FastSAM-3D | 2, 4, 6 | 8.24° ± 1.08° | 8.23° | 55 mm | 0.15° |
| FastSAM-3D | 2, 4 | 8.24° ± 1.06° | 8.21° | 47 mm | 0.17° |
| FastSAM-3D | 4, 6 | 8.78° ± 1.08° | 8.81° | 92 mm | 0.16° |
| FastSAM-3D | 2 | 8.97° ± 1.10° | 8.94° | 88 mm | 0.21° |

- Causal and aligned error now coincide (lag 0.4 frame for the headline), so the paper needs one error figure, not two. Every arm gains what the Butterworth's lag cost, about 0.24°; the ranking of arms and the gaps between them are those of section 0.
- Headline by task: Lifting 8.00°, Polishing 8.86°, Screwing 8.96°, SideOverhead 9.35°, RobotPolishing 7.73°, RobotWelding 7.88°.

**IK, horizon, filter and weights** (four cameras, mean fusion):

| Variant | Causal | Aligned | Lag (frames) | Jitter |
|---|---|---|---|---|
| MHE, horizon 7, tuned 1-euro (paper setting) | 8.46° ± 0.84° | 8.46° | 0.4 | 0.153° |
| Per-frame IK, same filter | 10.05° ± 1.58° | 10.03° | 1.0 | 0.145° |
| MHE, horizon 3 / 5 / 10 / 15 / 20 | 8.46 / 8.46 / 8.46 / 8.47 / 8.47° | 8.45 to 8.46° | 0.3 to 0.6 | 0.160 to 0.144° |
| MHE, no filter | 8.45° | 8.46° | −0.4 | 0.279° |
| MHE, 5 Hz Butterworth, order 4 (submitted) | 8.70° | 8.45° | 2.9 | 0.156° |
| MHE, state weight 1e-4 / 1e-2 | 9.46° ± 1.51° / 11.99° ± 1.06° | | 0.3 / 0.6 | 0.161 / 0.132° |
| MHE, control weight 1e-6 / 1e-4 | 8.46° / 8.46° | | 0.4 | 0.153° |

- The three-way filter comparison for the paper: none 8.45°, jitter 0.279°; 5 Hz Butterworth 8.70°, jitter 0.156°, lag 2.9 frames; tuned 1-euro 8.46°, jitter 0.153°, lag 0.4 frame.
- The 1-euro grid (20 points) is flat around the chosen point: causal error 8.46° to 8.84°, the worst being the slowest setting (0.5 Hz, beta 0.5; lag 3.8 frames).
- Horizon length has no effect on accuracy with this filter (all within 0.01°); a longer horizon lowers jitter slightly.

**Fusion rules:** mean 8.46°, inverse variance 8.57°, reprojection weights 8.94°, with symmetry refinement 9.59°.

**Statistics**, on participant means, Holm within each family, equivalence at ±1°:

- Arms (10). Proposed against mmpose: −4.11° [−4.68, −3.54], p < 0.001. 3D fusion against 2D triangulation, four cameras: −0.87° [−1.28, −0.45], p = 0.002. Camera ladder: 1 against 2 same-side +0.32°, 2 same-side against 4 +0.52°, 1 against 4 +0.84°, same-side against opposed pair +0.30°, all p ≤ 0.006. FastSAM-3D against NLF-S on cameras 2 / 4-6 / 2-4 / 2-4-6: −0.37° / −0.54° / −0.46° / −0.42°, none significant (Holm p = 0.27 each); equivalence within ±1° not established after correction (Holm p = 0.11 to 0.14).
- IK and filter (14). MHE against per-frame IK: +1.59° [1.11, 2.07], p < 0.001, not equivalent. Every horizon equivalent to 7 (largest difference 0.009°). No filter: −0.007°, equivalent. 5 Hz Butterworth: +0.24° [0.22, 0.26], p < 0.001, equivalent within ±1°.
- MHE weights (4). State weight 1e-4: +1.00° [0.56, 1.44], p = 0.002; 1e-2: +3.53°, p < 0.001; control weight: equivalent (differences below 0.003°).
- Fusion against the mean (3). Inverse variance +0.11°, p = 1.0, equivalent; reprojection weights +0.48° [0.11, 0.84], p = 0.028, equivalent; with symmetry refinement +1.13° [0.66, 1.59], p < 0.001.

No conclusion of section 0 changes.

**REBA**, headline arm, causal (absolute / reference's neutral / own neutral): mean absolute error 1.05 / 0.89 / 0.70, same risk level 59 % / 70 % / 75 %, weighted kappa 0.47 / 0.62 / 0.71. High-risk frames rated lower, absolute angles: 45 % ± 21 % by participant, 56 % pooled. Within 0.02 of the Butterworth figures; the conclusions of section 5 stand.

**Hand-to-robot distance**, headline arm, causal, all robot trials pooled: left hand bias −11 mm, SD 38 mm, 99th percentile of absolute error 134 mm; right hand +7 mm, 33 mm, 113 mm. Close frames missed at the 105 mm threshold: 42 % (left), 58 % (right); at 300 mm: below 0.1 % for both, with 8.6 % (left) and 0.3 % (right) false alarms. The open decision of section 6 is unchanged.

**Not regenerated with the 1-euro filter:** the TensorRT and tiled-detection replays (interchangeable with TorchScript to 0.05° and 0.21° per trial with the Butterworth; the filter acts after inference) and the fusion-validation study of ticket 10 (it compares fusion rules under one common filter).

## 0c. OpenCap LabValidation (ticket 16), 2026-11-15

The paper pipeline, unchanged (NLF-S, mean fusion, tuned 1-euro filter, MHE horizon 7, landmark map of fold 0), on the nine subjects of OpenCap's laboratory validation set that have videos: 144 trials at 60 Hz, reference from the dataset's mocap markers through our model and IK. Method, tables and limits in `docs/research/opencap-evaluation.md`; outputs in `/root/workspace/oof/t16`.

| Arm | Causal | Marker error | Squat | Sit-to-stand | Drop jump | Walking |
|---|---|---|---|---|---|---|
| 5 cameras | 8.8° ± 1.0° | 58 mm | 8.4° | 8.6° | 10.4° | 7.4° |
| 3 cameras | 9.3° ± 1.0° | 71 mm | | | | |
| 2 cameras (±45°) | 9.2° ± 1.0° | 68 mm | 8.6° | 9.0° | 10.6° | 8.0° |

- 26 scored DoF: the paper's 29 without the neck, which the dataset's markers do not observe. No solver failure; 7 dropped frames in the two-camera arm. Lag 0.3 frame, so causal and aligned error coincide. COMFI, four cameras: 8.46° ± 0.84°.
- Forearm pronation is not observed by the video pipeline, on either dataset. The twist between NLF's elbow and wrist landmarks is nearly constant (SD 3° to 10° within a trial, correlation 0.1 to 0.3 with the mocap's), so the estimated pronation stays at its calibration value, 66° to 70°, whatever the forearm does. On COMFI the reference moves around that value (56° to 91° by task): pronation RMSE 27.9° ± 9.2°, no constant offset. On OpenCap the reference is calibrated about 25° lower in 8 of 9 subjects (36° to 53° against 58° to 76° in the static trial) and stays there: RMSE 30°, seen as a +28° offset. Without the two pronation DoF: COMFI about 7.0° (from 8.46°), OpenCap 7.05° (five cameras) and 7.41° (two). OpenCap's own video results have the same weakness (pronation RMSE 16° to 22° against its mocap). **Decided (2026-11-15):** the 29 DoF are kept; the paper states that forearm pronation is not observed from video and gives the error with and without it. COMFI, headline arm, from the per-DoF scores of the existing runs: 8.46° ± 0.84° on 29 DoF, 7.02° ± 0.75° without the two pronation DoF (left 33.0° ± 17.9°, right 22.7° ± 5.5°). The next largest are cervical flexion (19.0°, a constant offset of −18.5°) and shoulder rotation (11.3° and 11.2°); every other DoF is below 10°.
- OpenCap's released angles against its own mocap angles, recomputed from its files on the same trials (HRNet, 2 / 5 cameras): 5.5° / 5.3° RMSE on its 18 lower-body and trunk rotations, 14.3° / 13.7° on its 10 arm rotations.
- On the 11 DoF both models have (ours against our reference, OpenCap's against its own): ours 6.6° (five cameras) and 7.0° (two), OpenCap HRNet 8.0° and 8.4°. Without shoulder and elbow: ours 5.9° and 6.2°, OpenCap 5.5° and 5.7°. So: comparable on the lower body and trunk, ours lower on the arm. The models and references differ, and OpenCap's processing is offline; ours is causal.
- The released video clips are not frame-synchronous (cameras 8 to 10 frames apart on average, each clip cut at a key frame). The evaluation synchronises the cameras from the body's motion and sets the video clock on the mocap's per trial, as on COMFI; the paper must say so.
- In 6 walking trials the frontal camera follows a bystander; a view whose person is more than 1 m from the other views' is treated as empty.
- NLF under-sizes tall subjects (10 % at 1.96 m); marker error follows height, joint error does not.

## 0d. Update of 2026-11-15: reference head point, decisions, current figures

**These are the current figures.** They supersede sections 0 and 0b wherever a number differs; the differences between arms are those of section 0b to 0.03°.

**Neck flexion.** The estimate tracked neck flexion (correlation 0.9 with the reference, 4° of scatter) with a constant offset of −18.5° (−8° to −30° by participant), present from the first frame. Cause: the model takes the head's vertical axis from the shoulders to a Head point at the top of the head; the reference used the centroid of the mocap head band, about 8 cm lower, which tilts that axis forward. The reference's Head is now the band's centroid raised along the band's normal by the height of the NLF head-top landmark above it, measured on each fold's fitting participants (77 to 83 mm; `fit_head_raise.py`, commit `7b6c116`). The reference was regenerated (108 trials, no failure) and every run rescored; no estimate was rerun. Neck flexion error: 19.0° to 10.5° ± 5.4° (offset −8.7°, −21° to +2° by participant: the band is not worn the same way by everyone). Reference in `/root/workspace/oof/ref28`, tables in `/root/workspace/oof/t28`.

**Arms, 18 participants, 108 trials each, no solver failure or dropped frame.**

| Arm | Cameras | Error | Marker error | Jitter |
|---|---|---|---|---|
| NLF-S, 3D fused (headline) | 0, 2, 4, 6 | 8.16° ± 0.82° | 50 mm | 0.15° |
| NLF-S, 3D fused | 0, 4 (opposed) | 8.38° ± 0.86° | 58 mm | 0.19° |
| NLF-S, 3D fused | 0, 2 (same side) | 8.69° ± 0.88° | 104 mm | 0.19° |
| NLF-S, 3D fused | 0 | 9.03° ± 0.94° | 115 mm | 0.24° |
| NLF-S, 2D triangulated | 0, 2, 4, 6 | 9.05° ± 0.98° | 69 mm | 0.22° |
| NLF-S, 2D triangulated | 0, 2 | 12.54° ± 1.28° | 93 mm | 0.49° |
| NLF-S, 2D triangulated | 0, 4 | 22.11° ± 8.35° | 101 mm | 0.28° |
| RTMPose+LSTM (the baseline of the submitted paper) | 0, 2, 4, 6 | 12.34° ± 1.05° | 76 mm | 0.19° |
| NLF-S, 3D fused | 2, 4, 6 / 2, 4 / 4, 6 / 2 | 8.36° / 8.40° / 9.03° / 9.06° | 60 / 53 / 105 / 100 mm | |
| FastSAM-3D | 2, 4, 6 / 2, 4 / 4, 6 / 2 | 7.94° / 7.93° / 8.48° / 8.70° | 53 / 46 / 91 / 86 mm | |

- Headline without the two forearm pronation DoF: 6.69° ± 0.70°. Pronation: 33.2° (left), 22.8° (right). Besides pronation and neck flexion, only shoulder rotation exceeds 10° (11.1° both sides).
- By task: Lifting 7.72°, Polishing 8.54°, Screwing 8.70°, SideOverhead 9.00°, RobotPolishing 7.42°, RobotWelding 7.59°.
- Causal and aligned error coincide to 0.01° (lag 0.4 frame).

**IK, filter, weights, fusion, inference.**

| Variant | Error | Lag (frames) | Jitter |
|---|---|---|---|
| MHE, horizon 7, tuned 1-euro (paper setting) | 8.16° ± 0.82° | 0.4 | 0.153° |
| Per-frame IK, same filter | 9.73° ± 1.54° | 1.0 | 0.145° |
| MHE, horizon 3 / 5 / 10 / 15 / 20 | 8.16 / 8.16 / 8.16 / 8.17 / 8.17° | 0.3 to 0.6 | 0.160 to 0.144° |
| MHE, no filter | 8.16° | −0.4 | 0.279° |
| MHE, 5 Hz Butterworth, order 4 (submitted) | 8.40° | 2.9 | 0.156° |
| MHE, state weight 1e-4 / 1e-2 | 9.14° / 11.70° | | |
| MHE, control weight 1e-6 / 1e-4 | 8.16° / 8.16° | | |
| Fusion: mean / inverse variance / reprojection weights / with symmetry | 8.16 / 8.27 / 8.67 / 9.32° | | |
| TensorRT, per-view detection (1-euro) | 8.16° ± 0.82° | 0.5 | 0.153° |
| TensorRT, tiled detection (1-euro) | 8.16° ± 0.81° | 0.4 | 0.153° |

TensorRT and tiled detection differ from TorchScript by at most 0.05° and 0.25° on one trial.

**Statistics**, participant means, Holm within each family, equivalence at ±1°:

- Arms (10). Proposed against mmpose: −4.18° [−4.74, −3.63], p < 0.001. 3D fusion against 2D triangulation: −0.89° [−1.30, −0.47], p = 0.001. Camera ladder: 1 against 2 same-side +0.34°, 2 same-side against 4 +0.53°, 1 against 4 +0.87°, same-side against opposed pair +0.31°, all p ≤ 0.003. FastSAM-3D against NLF-S on cameras 2 / 4-6 / 2-4 / 2-4-6: −0.36° / −0.55° / −0.47° / −0.42°, none significant (Holm p = 0.24 each); equivalence not established (Holm p = 0.10 to 0.18).
- IK and filter (14). MHE against per-frame IK: +1.57° [1.09, 2.05], p < 0.001. Every horizon equivalent to 7 (largest difference 0.01°). No filter: −0.01°, equivalent. 5 Hz Butterworth: +0.24° [0.22, 0.26], p = 0.002, equivalent.
- MHE weights (4). State weight 1e-4: +0.98° [0.54, 1.43], p = 0.002; 1e-2: +3.54°, p < 0.001; control weight equivalent.
- Fusion against the mean (3). Inverse variance +0.11°, p = 1.0, equivalent; reprojection weights +0.50° [0.14, 0.87], p = 0.024, equivalent; with symmetry +1.16°, p < 0.001.

**Fusion robustness (ticket 10), rerun with the 1-euro filter and this reference.** Held-out participants (12), mean / inverse variance / median: clean 8.07 / 8.13 / 8.17°; views masked half the time 8.68 / 8.75 / 8.87°. Validation participants (6), a 1 m outlier on one limb of one view: 11.37 / 11.62 / 8.59°. The uncertainty-weighted median is unusable (12° to 17°). Conclusions of ticket 10 unchanged. Tables in `/root/workspace/oof/t28/robustness`.

**REBA**, headline arm (absolute / reference's neutral / own neutral): mean absolute error 0.91 / 0.82 / 0.71, same risk level 63 % / 72 % / 75 %, weighted kappa 0.53 / 0.66 / 0.70. Absolute angles gain from the neck correction (1.05 and 59 % before); own neutral is unchanged. High-risk frames rated lower: 46 % / 51 % / 54 % by participant, so the neck offset was not the cause of the underestimation.

**Hand-to-robot distance:** unchanged (SD 33 to 38 mm; at 300 mm, under 0.1 % of close frames missed, false alarms 8.6 % left and 0.3 % right).

**Decisions of 2026-11-15.**

- REBA: the paper's main figure is each side relative to its own neutral posture (0.71, 75 %, kappa 0.70).
- Hand-to-robot distance: the claim is reworded at a 300 mm threshold.
- FastSAM-3D: one sentence (another foundation model under the same protocol, slightly more accurate, much slower). Participant 3361's relabelled meshes need no further action.
- Synchronisation, COMFI: hardware between cameras and mocap, software between the webcams; details as in the submitted paper.
- COMETH: one sentence, no experiment.
- InstantHMR (medium, fit on 1 crop) was tried as a front end and dropped: same accuracy as NLF-S on one COMFI participant (9.01° against 9.12°), 0.3° to 0.9° behind on two OpenCap subjects, same speed once both are batched in TensorRT (7.6 against 8.8 ms per four-view frame), no gain on pronation, and landmark maps fitted on its own meshes did not help. Not in the paper.
- OpenCap is unaffected by the head point (its reference has no head markers).

## 0e. Timing and latency (ticket 19), 2026-11-15

Workstation (RTX 4500 Ada, 32 cores, idle), frames released at the camera rate, every stage waiting for the GPU. Full tables: `docs/research/timing-workstation.md`; procedure: `docs/research/timing.md`; per-frame files in `/root/workspace/oof/t19`.

| Dataset | Inference | Processing per frame, median (95th, 99th) | Rate | Frames late for the next one | Latency of a motion event, median (95th, 99th) |
|---|---|---|---|---|---|
| COMFI, 4 cameras, 40 Hz | TorchScript, per-view detection | 22.7 ms (25.9, 30.9) | 44 fps | 17.4 % | 46 ms (59, 73) |
| COMFI | TensorRT, per-view detection | 17.5 ms (20.1, 24.8) | 57 fps | 1.07 % | 40 ms (52, 54) |
| COMFI | TensorRT, tiled detection | 14.2 ms (16.8, 20.7) | 70 fps | 0.25 % | 37 ms (48, 50) |
| OpenCap, 4 cameras, 60 Hz | TorchScript, per-view detection | 22.9 ms | 44 fps | all | not sustained |
| OpenCap | TensorRT, per-view detection | 18.4 ms | 54 fps | all | not sustained |
| OpenCap | TensorRT, tiled detection | 14.8 ms (17.6, 21.9) | 67 fps | 16.9 % | 29 ms (37, 43) |

- COMFI: 6 trials of participant 1012, 14,991 frames per configuration. OpenCap: subject 4, 16 trials, 4,293 frames, cameras 0, 1, 3, 4.
- Stages, COMFI, TensorRT with tiled detection (median): upload 1.8 ms, detection 3.2, NLF crop 2.3, NLF 2.0, fusion 0.14, filter 0.03, IK 3.5, output 0.16.
- Latency is the wait for the next frame (spread over one period), the delay from a frame's arrival to its joint angles (with the backlog a late frame leaves), and the lag of the estimate behind the reference (0.4 frame on COMFI, 0.3 on OpenCap). The cameras' acquisition delay and the synchronisation error between cameras are not in it. Rates are processing rates on recorded video: a live system is capped by its cameras, 40 Hz on COMFI.
- TorchScript, the default, does not hold 40 Hz on this workstation: one frame in six is late. The 40 Hz claim needs the TensorRT backend; with tiled detection there is margin (99th percentile 20.7 ms for 25 ms).
- At 60 Hz only TensorRT with tiled detection keeps up, with one frame in six late; the median processing time (14.8 ms) is under the 16.7 ms period.
- **IK cost**, recorded NLF output, same session: moving-horizon IK 2.80 ms per frame (95th 3.59), per-frame IK 1.54 ms (95th 1.81). The moving horizon costs 1.26 ms more (1.7 ms in the submitted paper).
- **COMFI timestamps**, 108 trials, 265,037 frames per camera: mean interval 25.07 ms. The timestamps are arrival times at the recording computer, not exposure times, and arrivals are bursty, so dropped frames can only be bounded: between 0.29 % (frames missing against 40 Hz) and 1.4 to 2.5 % (intervals longer than 1.5 periods, longest 83 ms, usually followed by shorter ones). The four cameras' times within a frame differ by 17.1 ms (median), 23.9 ms (95th percentile) and 27.8 ms (99th), 69 ms at most: the software synchronisation between webcams is about one camera period. OpenCap's videos carry no acquisition timestamps.
- The laptop (RTX 3050 Ti) and the RTX 4070 desktop remain to be measured by the author with the same procedure.

## 0f. Calibration (tickets 14, 15, 17), 2026-11-15

Full tables: `docs/research/calibration-comfi.md`, `docs/research/calibration-opencap.md`; scratch in `/root/workspace/oof/t15` and `t17`.

**In the toolbox.** Checkerboard, wand and human-pose calibration are now part of RT-COSMIK (`rtcosmik.calibration`, `scripts/python/calibration/`); the separate repository is no longer needed.

**Method.** The human-pose calibration registers each camera's metric 3D landmarks rigidly onto the reference camera's, over a recording of one person walking. It replaces the essential-matrix method of the calibration repository: on the 18 circular walks the two are equally accurate, but on straight walks the essential matrix fails for the adjacent pair (6.0° and 340 mm against 0.66° and 42 mm). It needs no 2D keypoints and no stature. A recording in which the body centre stays within 0.5 m is refused: none of the 36 walks, 102 of the 108 evaluation-task recordings.

**Protocol (COMFI).** For each of the 18 participants, NLF with the participant's out-of-fold map on the CircularWalking recording (2,100 to 3,600 frames, not an evaluation task), cameras 0, 2, 4, 6, camera 0 as reference. Errors are those of each camera's pose relative to camera 0 against the dataset calibration. The six evaluation tasks are then replayed from the recorded NLF output with these extrinsics; camera 0 keeps its dataset world pose, so the comparison isolates the error between cameras.

| Pair | Baseline | Rotation, median (quartiles; max) | Camera position | Baseline length |
|---|---|---|---|---|
| Adjacent (camera 2), n = 18 | 0.81 m | 0.84° (0.62, 1.27; 1.99) | 40 mm (29, 58; 113) | 4.3 % |
| Opposite (cameras 4, 6), n = 36 | 5.34 m | 1.58° (0.92, 2.15; 3.62) | 195 mm (128, 239; 317) | 2.9 % |

**Downstream (COMFI, 108 trials).**

| Calibration | Joint angles | Markers | Hand-to-robot distance, SD (left, right) |
|---|---|---|---|
| Dataset | 8.16° | 50 mm | 38, 33 mm |
| Human pose | 8.16° (−0.004°; largest participant difference 0.03°) | 98 mm | 52, 48 mm |

- Joint angles do not depend on the calibration. Positions do: the error doubles, to the level of one camera or two cameras on the same side (115 and 104 mm). Registering every view onto the reference camera makes the views agree at that camera's own estimate, so its depth bias is no longer averaged out by the opposite cameras.
- At 300 mm the hand-distance check keeps under 0.3 % of missed frames; false alarms on the left hand go from 8.6 % to 12.0 %.
- The essential-matrix method gives the same positions (99 mm); with the scale set from stature, 71 mm on average but 144 mm for the worst participant.

**Sensitivity to extrinsic errors (COMFI, 108 trials, cameras 2, 4, 6 perturbed in random directions, seeded per trial).**

| Perturbation | Joint angles | Markers |
|---|---|---|
| none | 8.16° | 50.0 mm |
| rotation 0.5°, 1°, 2°, 5° | 8.16° throughout | 50.5, 52.6, 59.0, 93.6 mm |
| shift 10, 20, 50, 100 mm | 8.16° throughout | 50.3, 50.5, 53.9, 62.9 mm |

Joint angles change by at most 0.02° for any participant. Positions tolerate 1° and 50 mm (under 4 mm of extra error); hand-distance SD stays within 2 mm of the unperturbed value up to 1° and 20 mm.

**OpenCap (ticket 17).** The dataset calibration is a checkerboard, so the OpenCap results of section 0c already are results under a deployment calibration. Human-pose calibration was computed for each walking trial from the subject's five other walking clips (about 430 frames, 7 s), nine subjects. The clips cover a short stretch (travel 0.43 to 0.50 m), below the 0.5 m rule, which was lifted for this measurement; the squat, sit-to-stand and most drop-jump trials are refused.

| | Checkerboard | Human pose |
|---|---|---|
| Extrinsics relative to Cam0 (median over subjects) | reference | 2.0° to 4.0°, 106 to 344 mm on baselines of 1.5 to 6.0 m |
| Walking, 5 cameras: joint angles | 7.45° | 7.47° |
| Walking, 3 cameras | 8.01° | 8.04° |
| Walking, 2 cameras | 7.95° | 7.98° |
| Walking, 5 cameras: markers | 53.5 mm | 175 mm (62 mm with the frame refitted) |

Markers on OpenCap are compared through one rigid transform per session to the mocap frame; with the transform fitted under the checkerboard calibration the human-pose calibration shows its global offset, with the transform refitted it shows what is left (62 mm against 53.5).

**Decision (story 36).** No change of method. The cost on joint angles is nil on both datasets, and the cost on positions comes from the monocular depth bias of the reference view, which no calibration from the same landmarks removes (the 2D-based alternative gives the same positions). For the paper: human-pose calibration is validated for joint angles and ergonomic scoring; position-critical uses (robot distances) need the checkerboard calibration, whose pipeline results are those of OpenCap. Not validated: the checkerboard and wand procedure on COMFI itself (no checkerboard images of that rig exist, and no new recordings are made).

## 1. Headline

| Configuration | Causal | Aligned |
|---|---|---|
| Submitted map (fitted on the evaluated participants), same scoring | 9.98° ± 1.08° | 9.74° ± 1.10° |
| Out-of-fold maps, inverse-variance fusion | 9.14° ± 1.08° | 8.90° ± 1.10° |
| Out-of-fold maps, unweighted mean fusion | 8.82° ± 0.86° | 8.57° |

- 108/108 trials, 0 solver failures, 0 dropped frames, 0 joint-limit violations.
- Pipeline lag with the 5 Hz filter: 2.9 frames at 40 Hz. Marker error 55 mm.
- Fold means of causal error (inverse variance): 10.18°, 8.76°, 8.78°, 8.64°, 9.54°, 8.91°. Fold 0 is the hardest group; not investigated.
- The medial ankle landmarks keep their hand-picked vertices, like the pelvis: their candidate vertices tie on marker error, and the ankle frame is sensitive to the choice.

**To decide:** which configuration is the headline (see 2 and 3).

## 2. Filter and IK studies (inverse-variance fusion)

| Variant | Causal | Aligned | Lag (frames) | Jitter |
|---|---|---|---|---|
| MHE, horizon 7, 5 Hz order 4 (paper setting) | 9.14° | 8.90° | 2.9 | 0.173° |
| Per-frame IK, same filter | 10.26° | 9.99° | 3.4 | 0.169° |
| MHE, horizon 3 / 5 / 10 / 15 / 20 | 9.12 / 9.13 / 9.11 / 9.11 / 9.04° | 8.91 to 8.77° | 2.6 to 3.0 | 0.183 to 0.163° |
| MHE, no filter | 8.90° | 8.90° | −0.4 | 0.318° |
| MHE, 3 / 4 / 8 / 10 Hz, order 4 | 9.51 / 9.26 / 8.98 / 8.94° | 8.90° | 5.4 / 3.8 / 1.5 / 1.0 | 0.146 / 0.161 / 0.208 / 0.229° |
| MHE, order 2 at 6 / 8 / 10 Hz | 8.95 / 8.92 / 8.91° | 8.90° | 1.1 / 0.6 / 0.4 | 0.188 / 0.210 / 0.229° |

Statistics (ticket 11: Holm within the family of 14 comparisons, equivalence by two one-sided tests at ±1°):

- MHE against per-frame IK: +1.13° [0.64, 1.62], Holm p = 0.002, not equivalent. This claim is firm.
- Horizon length: every horizon from 3 to 20 is equivalent to 7 within ±1°. Some are also significantly different, by 0.01° to 0.03°. Wording: "equivalent within ±1°", not "no difference".
- The filter does not change aligned error. It trades lag (hence causal error) against jitter.
- The ±1° margin is wide for these effects: it declares a 0.38° degradation (3 Hz) equivalent.

**To decide:** ticket 09 planned the headline as "MHE without filter unless a 1€ filter has lower causal error", to be fixed before seeing out-of-fold results. That was not recorded in time. The numbers above use the 5 Hz filter.

## 3. Fusion rule

| Rule | Causal | Difference to inverse variance | Better for | Marker error | Jitter | Cost per frame |
|---|---|---|---|---|---|---|
| Inverse variance (submitted) | 9.14° ± 1.08° | | | 55.1 mm | 0.173° | 0.04 ms |
| Unweighted mean | 8.82° ± 0.86° | −0.32° [−0.61, −0.03] | 9 of 18 | 55.3 mm | 0.156° | 0.04 ms |
| Reprojection weights (Bragagnolo et al.) | 9.39° ± 1.07° | +0.25° [−0.08, +0.58] | 6 of 18 | 63.1 mm | 0.235° | 0.26 ms |
| Reprojection weights + symmetry refinement (Bragagnolo et al.) | 10.10° ± 1.26° | +0.96° [+0.51, +1.41] | 4 of 18 | 74.8 mm | 0.218° | 7.1 ms |

- Mean against inverse variance: the t-based interval excludes zero, the signed-rank test gives p = 0.18 (0.31 after Holm). Equivalent within ±1°. Almost all of the gain is on SideOverhead (10.49° against 11.88°). Fewer shoulder flips (0.25 % of frames against 0.87 %).
- The full Bragagnolo method is significantly worse by about 1° and costs 7 ms per frame in our implementation.
- The data do not show that weighting by NLF's uncertainty helps.
- On hand-to-robot distance the mean has heavier tails than inverse variance (see 6). On REBA the two are within a point or two (see 5).

**Planned:** unweighted mean as the headline rule, with one sentence and a table for the three others. Supported wording: "performed at least as well as inverse-variance weighting and better than the two reprojection-based rules, at the lowest cost". "Best of four" is stronger than the tests. The studies of section 2 are to be replayed with the mean so that every table uses the headline rule. The method section and any claim built on NLF's uncertainty must change.

## 4. Speed: TensorRT and tiled detection

Workstation GPU (RTX 4500 Ada), 1012 Lifting, 400 frames, median ms per frame, idle machine.

| Views | TorchScript, per-view detection | TensorRT, per-view detection | TensorRT, tiled detection |
|---|---|---|---|
| 1 | 13.9 (72 fps) | 9.9 (101 fps) | 9.9 (101 fps) |
| 2 | 18.2 (55 fps) | 12.9 (78 fps) | 12.0 (84 fps) |
| 3 | 21.7 (46 fps) | 15.8 (63 fps) | 13.5 (74 fps) |
| 4 | 24.3 (41 fps) | 18.5 (54 fps) | 15.1 (66 fps) |

Four views, per stage: detection 7.8 / 7.6 / 3.7 ms; image upload 1.7 / 1.7 / 2.4; crop and NLF 9.5 / 4.4 / 4.3; fusion 0.5; IK 3.5.

Accuracy against mocap:

- TensorRT against TorchScript, 108 trials: causal error 9.09° against 9.10° on 17 complete participants at the time of writing; largest difference on one trial 0.06°. Landmarks differ by 1.55 mm median on identical boxes; TorchScript differs from itself by 2 mm when one optimization flag changes.
- Tiled detection (detector image 736 x 1280): against per-view detection on one trial, joint angles differ by 0.18° median and 4.0° at most. The 108-trial pass against mocap is running.

**Planned:** report TorchScript for accuracy and TensorRT for speed, the two being interchangeable. Tiled detection only if its pass confirms accuracy. Laptop timings (RTX 3050 Ti, RTX 4070) at the very end, from a clean install.

CPU: NLF-S through ONNX Runtime on an i9-14900K takes about 34 ms per crop on 4 threads and 23 ms on 8 (rough test, machine not idle, no OpenVINO). Real time on CPU is unlikely with NLF-S at several views. Tickets 24 and 25 (InstantHMR, CPU) come after the resubmission tickets.

## 5. REBA (ticket 12)

Causal, 18 participants. Lag is not the cause of disagreement: aligned figures are almost identical.

| Fusion | Angle reference | Score bias | Mean abs. error | Same risk level | Weighted kappa |
|---|---|---|---|---|---|
| Inverse variance | absolute | −0.58 | 1.03 ± 0.19 | 61 % | 0.49 |
| Inverse variance | reference's neutral | +0.13 | 0.94 ± 0.19 | 68 % | 0.59 |
| Inverse variance | own neutral | −0.25 | 0.74 ± 0.15 | 73 % | 0.68 |
| Unweighted mean | absolute | −0.73 | 1.07 ± 0.22 | 59 % | 0.47 |
| Unweighted mean | reference's neutral | +0.05 | 0.91 ± 0.18 | 69 % | 0.61 |
| Unweighted mean | own neutral | −0.29 | 0.72 ± 0.12 | 75 % | 0.69 |

Frames the reference rates high risk:

| Fusion | Angle reference | Rated lower by the system | Score error there |
|---|---|---|---|
| Inverse variance | absolute | 39 % ± 21 % | −1.13 |
| Inverse variance | own neutral | 53 % ± 23 % | −1.16 |
| Unweighted mean | absolute | 47 % ± 20 % | −1.27 |
| Unweighted mean | own neutral | 55 % ± 21 % | −1.15 |

- About half of the high-risk frames are rated one level down, almost all to "medium". Neither series reaches "very high".
- Confusion matrix (inverse variance, own neutral), share of each reference level rated correctly: negligible 92 %, low 70 %, medium 74 %, high 44 %.
- Postures selected on the estimate (what a user sees: the highest score held for 0.5 s) are slightly overstated against the reference (+0.1 to +0.9 points); frames selected where the reference is high are understated (−1.1 to −1.3). A figure must say which selection it uses.
- Absolute mode does not score shoulder raise (it is a distance with no absolute zero), so it has fewer high-risk frames and is not directly comparable with the neutral-referenced rows.

**To decide:** how to state the underestimation of high-risk postures as a limitation; which angle reference is the main one (own neutral is what a deployed system would use).

## 6. Hand-to-robot distance (ticket 13)

33 of the 36 robot trials (COMFI's robot-state files are empty for 1602 RobotWelding, 1847 RobotPolishing, 2307 RobotWelding), about 84,000 frames per hand, causal. Error is estimate minus reference: positive means the hand is estimated farther from the robot than it is.

| Fusion | Hand | Bias | SD | 95th pct abs. | 99th pct abs. |
|---|---|---|---|---|---|
| Inverse variance | left | −2.5 mm | 40.4 mm | 81 mm | 113 mm |
| Inverse variance | right | +7.9 mm | 33.0 mm | 74 mm | 113 mm |
| Unweighted mean | left | −10.8 mm | 39.6 mm | 85 mm | 136 mm |
| Unweighted mean | right | +7.6 mm | 37.8 mm | 85 mm | 125 mm |

Close-approach frames missed (reference below the threshold, estimate above), inverse variance, left / right hand: 105 mm: 58.5 % / 58.2 %; 150 mm: 24.1 % / 34.7 %; 200 mm: 7.8 % / 6.2 %; 300 mm: 0.2 % / 0.2 %.

- The submitted text's "as close as 105 mm" does not match these trials: the reference hand comes as close as 63 mm and is below 105 mm for 16 to 20 % of the time, because the participant holds the tool. How the 105 mm figure was obtained has not been checked.
- Per-trial bias ranges from about −56 to +75 mm.
- Approaches (reference below the threshold for at least 5 frames, 125 ms), inverse variance, left / right hand: at 105 mm, 37 / 37 approaches, 12 / 12 never detected, 13 / 15 detected late (median delay 2.4 s / 0.35 s, 95th percentile 13 s / 21 s); at 200 mm, 88 / 69 approaches, 1 / 2 never detected, late ones after 0.2 s / 0.1 s (median); at 300 mm, 49 / 34 approaches, none missed, detected within 75 to 100 ms (median) and 135 to 175 ms (95th percentile).
- Proposed rule, written before the out-of-fold numbers were computed: keep the application claim only if, for both hands, the 99th percentile of absolute error is below 105 mm, the 95th percentile of overestimation is below 52.5 mm, and the miss rate at 105 mm is below 5 %. Every arm fails it. The miss-rate criterion fails by a factor of ten; the 99th-percentile criterion by 8 mm for inverse variance.
- Measured end-to-end latency (ticket 19) is not included.

**To decide:** drop the application, or reword it as proximity monitoring at a 300 mm scale with these error and miss rates. The rule's limits are a proposal to confirm.

## 7. FastSAM-3D out of fold (ticket 06)

Markers read from the stored meshes at each fold's map. Camera-0 meshes are not stored, so there is no four-camera configuration. 108 trials per configuration, 0 solver failures.

| Configuration | Cameras | Causal | Aligned | Marker error (causal) |
|---|---|---|---|---|
| 1 camera | 2 | 9.71° ± 1.28° | 9.41° | 90 mm |
| 2 cameras, same side | 4, 6 | 9.26° ± 1.27° | 9.07° | 94 mm |
| 2 cameras, opposed | 2, 4 | 8.80° ± 1.24° | 8.50° | 51 mm |
| 3 cameras | 2, 4, 6 | 8.77° ± 1.26° | 8.50° | 58 mm |

- For reference, NLF-S with four cameras: 9.14° (inverse variance), 8.82° (unweighted mean). Not paired comparisons: the NLF arms at cameras 2, 2-4, 4-6 and 2-4-6 come from replays (ticket 08).
- 3361's stored labels 2, 4, 6 hold calibrated cameras 6, 0, 2 (checked against the reference), so it runs the mirror image of each configuration, which is the same geometry on this rig. n stays 18.
- One trial (1508 Polishing, cameras 4-6) has an alignment lag of 61 frames: the knee-flexion lock fails there (correlation 0.29). Its causal error is unaffected (8.50°); its aligned error (11.44°) is not meaningful.
- FastSAM-3D remains an offline arm (about 390 ms per view).

**To decide:** keep 3361 with mirrored cameras (n = 18) or drop it from this arm (n = 17).

## 8. Still to come

- NLF arms by replay: camera configurations, NLF-2D triangulation, MHE weights (ticket 08); 1€ filter (09); fusion validation, including whether NLF's uncertainty is reliable (10).
- Arm-level statistics (ticket 11 code is ready; it needs the NLF arms of ticket 08).
- Toolbox simplification (18). Calibration (14, 15, 17) is in section 0f, timing (19) in 0e.
- To confirm with the COMFI authors: how COMFI synchronizes mocap and video in hardware.
