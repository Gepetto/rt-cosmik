# Resubmission findings to discuss before rewriting the paper (ticket 21)

State on 2026-11-13. Every number is on COMFI, 18 participants, six tasks, 108 trials, four cameras (0, 2, 4, 6), out-of-fold landmark maps, participant mean ± SD unless stated. "Causal" error compares estimate and reference at the same frame after removing only the per-trial clock offset between mocap and video (2 or 3 frames). "Aligned" error first removes the lag estimated on knee flexion, as the submitted paper did.

Where the data are: runs in `/root/workspace/oof/runs/<id>/<task>/<arm>/`, tables in `results/oof/` of the main checkout and in `/root/workspace/oof/t11`, `t12`, `t13` (not in git).

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
- Calibration in the toolbox, human-pose calibration on COMFI and OpenCap (14 to 17); toolbox simplification (18); sustained timing (19).
- To confirm with the COMFI authors: how COMFI synchronizes mocap and video in hardware.
