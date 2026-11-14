# Fusion validation (ticket 10)

State on 2026-11-14, branch `ticket-10`. COMFI, 18 participants, six tasks, 108 trials, cameras 0, 2, 4, 6, out-of-fold landmark maps, recorded NLF-S views replayed (`/root/workspace/oof/views`). No new recording. Outputs: `/root/workspace/oof/t10/reliability` and `/root/workspace/oof/t10/robustness` (tables), `/root/workspace/oof/t10/runs` (run folders, tags `robust_<condition>_<rule>`), commit `efd5a54`, reference from `/root/workspace/oof/runs26`.

## 1. Fusion rules and what each does with bad input

`settings.fusion` selects the rule; all four fuse, per landmark, the views' 3D positions expressed in the reference camera frame (`reconstruct_3d`). The default stays `mean`.

| Rule | Estimate per landmark | Reads sigma |
|---|---|---|
| `mean` (default) | unweighted mean of the views | no |
| `invvar` | mean weighted by `1 / sigma^2` | yes |
| `median` | unweighted median, per axis of the reference frame; with an even number of views, the mean of the two middle values | no |
| `wmedian` | weighted median per axis, weights `1 / sigma^2`: the value at which the sorted views reach half the total weight (the two neighbours are averaged only on an exact tie) | yes |

Handling, common to the four rules unless stated:

- **A view that saw nobody, or whose pose is not finite**, has weight 0 for every landmark of that frame. The fusion uses the remaining views, down to one. A frame no view saw gives no landmarks and the frame is dropped (the IK is not called).
- **A missing or non-positive sigma** (`invvar`, `wmedian`) gives that view weight 0 for that landmark. If no view has a sigma for a landmark, all present views get weight 1 (the rule falls back to `mean` or `median`).
- **Extreme weights.** Sigma is floored at 1 mm, and weights are normalised so the largest is 1 per landmark. There is no upper cap on the ratio of weights: with `invvar` a view whose sigma is 3 times smaller than the others' carries 9 times their weight; with `wmedian` a view holding more than half of the total weight decides the result alone. In the 108 recordings sigma spans 14 mm to 1.27 m (1st to 99th percentile 18 to 101 mm, median 36 mm), so weight ratios above 30 between views are common.
- **Outliers.** `mean` and `invvar` have no protection: one view off by `d` moves the result by `d / 4` with four views (more with `invvar` if that view is confident). `median` ignores one outlying view of four and one of three; with two views it is their mean. `wmedian` ignores an outlying view as long as it holds less than half of the weight.
- **Efficiency.** With four views `median` averages two of them; `wmedian` with unequal weights returns a single view's coordinate, so it loses the averaging that cancels each view's depth error (see section 2, fused landmark error).
- The medians are taken per axis of the reference camera frame, so they are not invariant to the choice of reference camera. Not measured.

Synthetic checks in `tests/unit/triangulation/test_triangulation.py`: `median` and `wmedian` against a 0.5 m outlier view; `invvar` against `mean` with Gaussian views of known noise, 6.86 mm against 14.58 mm when sigma is the true noise (10, 10, 40, 40 mm), 13.34 mm against 10.0 mm when the same sigmas are reported for views of equal noise (closed forms, reproduced within 3 %).

## 2. Does NLF's sigma predict a view's error?

`scripts/python/paper/study_sigma_reliability.py`. Each view's landmark is taken to the world frame and compared with the mocap marker at the same instant (clock offset removed), for the 29 landmarks that have a marker. A landmark is a surface vertex and the marker sits nearby, so part of the error is a constant sigma cannot know. Reference reading (`trial`): the mean error vector over frames and views is removed **per trial and per landmark**, in the world frame; a view's own bias stays in its error. Also given: nothing removed (`raw`), each view's own mean removed (`view`), and the `trial` error split along and across the view's line of sight (`depth`, `lateral`). The body turns during a trial, so a constant world-frame offset is an approximation.

Columns: rank correlation of sigma with error over the frames of a trial, per view and landmark; rank correlation across the views of a frame (both centred on the frame mean, which is what weighting relies on); how often the view with the smallest (largest) sigma has the smallest (largest) error.

Participants (18): 1012 1118 1508 1602 1847 2112 2198 2307 3361 4162 4216 4279 4509 4612 4665 4687 4801 4827. Trials: 108. Mean ± SD over participants of each participant's mean over trials, landmarks and views.

### Rank correlation of sigma with error

| Tasks | Error | View error RMS (mm) | Over time, per view | Across views, per frame | Best view found | Worst view found |
|---|---|---|---|---|---|---|
| all tasks | raw | 114.1 ± 21.9 | 0.15 ± 0.15 | 0.16 ± 0.08 | 0.30 ± 0.04 | 0.32 ± 0.03 |
| all tasks | trial | 108.1 ± 24.8 | 0.13 ± 0.15 | 0.13 ± 0.06 | 0.29 ± 0.04 | 0.31 ± 0.03 |
| all tasks | view | 50.9 ± 5.2 | -0.01 ± 0.04 | 0.13 ± 0.03 | 0.29 ± 0.01 | 0.31 ± 0.01 |
| all tasks | depth | 95.9 ± 24.6 | 0.11 ± 0.15 | 0.11 ± 0.06 | 0.29 ± 0.04 | 0.30 ± 0.02 |
| all tasks | lateral | 46.0 ± 6.8 | 0.07 ± 0.05 | 0.06 ± 0.08 | 0.26 ± 0.04 | 0.29 ± 0.06 |
| robot tasks | raw | 108.2 ± 22.6 | 0.11 ± 0.22 | 0.16 ± 0.12 | 0.31 ± 0.07 | 0.31 ± 0.06 |
| robot tasks | trial | 101.6 ± 25.5 | 0.08 ± 0.22 | 0.11 ± 0.10 | 0.30 ± 0.07 | 0.30 ± 0.06 |
| robot tasks | view | 46.1 ± 7.5 | -0.14 ± 0.08 | 0.12 ± 0.04 | 0.29 ± 0.02 | 0.31 ± 0.03 |
| robot tasks | depth | 89.2 ± 25.2 | 0.06 ± 0.22 | 0.11 ± 0.10 | 0.30 ± 0.07 | 0.30 ± 0.05 |
| robot tasks | lateral | 44.6 ± 7.2 | 0.03 ± 0.06 | 0.00 ± 0.13 | 0.25 ± 0.06 | 0.26 ± 0.10 |
| other tasks | raw | 117.0 ± 23.3 | 0.17 ± 0.13 | 0.16 ± 0.07 | 0.30 ± 0.04 | 0.32 ± 0.03 |
| other tasks | trial | 111.4 ± 26.1 | 0.16 ± 0.14 | 0.14 ± 0.06 | 0.29 ± 0.04 | 0.31 ± 0.04 |
| other tasks | view | 53.3 ± 4.5 | 0.05 ± 0.05 | 0.14 ± 0.04 | 0.29 ± 0.02 | 0.31 ± 0.02 |
| other tasks | depth | 99.2 ± 26.0 | 0.13 ± 0.14 | 0.11 ± 0.06 | 0.28 ± 0.04 | 0.30 ± 0.03 |
| other tasks | lateral | 46.7 ± 7.1 | 0.09 ± 0.05 | 0.10 ± 0.09 | 0.27 ± 0.05 | 0.30 ± 0.06 |

Chance for the last two columns is 0.25.

### Per view (offset removed per trial)

| View | Sigma (mm) | Error RMS (mm) | Rho over time, all | robot | other |
|---|---|---|---|---|---|
| 0 | 37.3 ± 1.6 | 115.3 ± 26.4 | 0.15 ± 0.20 | 0.12 ± 0.29 | 0.16 ± 0.17 |
| 2 | 36.1 ± 1.5 | 101.8 ± 24.1 | 0.09 ± 0.12 | -0.02 ± 0.24 | 0.14 ± 0.09 |
| 4 | 37.2 ± 1.6 | 103.6 ± 25.7 | 0.15 ± 0.18 | 0.09 ± 0.26 | 0.18 ± 0.17 |
| 6 | 36.9 ± 1.9 | 111.9 ± 26.4 | 0.14 ± 0.19 | 0.12 ± 0.26 | 0.16 ± 0.18 |

### Per landmark region (offset removed per trial)

| Region | Sigma (mm) | View error RMS (mm) | Rho over time | Rho across views, all | robot | other | Fused error mean / invvar / median / wmedian (mm) |
|---|---|---|---|---|---|---|---|
| pelvis | 25.5 ± 2.9 | 106.6 ± 29.1 | 0.15 ± 0.20 | 0.13 ± 0.12 | 0.11 ± 0.20 | 0.14 ± 0.11 | 25.8 / 26.0 / 27.4 / 70.6 |
| trunk | 30.0 ± 2.6 | 103.2 ± 25.4 | 0.08 ± 0.18 | 0.09 ± 0.13 | 0.09 ± 0.20 | 0.09 ± 0.13 | 26.4 / 27.0 / 28.2 / 63.2 |
| elbow | 44.9 ± 2.4 | 114.0 ± 24.6 | 0.15 ± 0.15 | 0.17 ± 0.07 | 0.11 ± 0.09 | 0.20 ± 0.08 | 39.9 / 39.4 / 41.4 / 61.4 |
| wrist | 55.8 ± 3.2 | 121.6 ± 21.5 | 0.17 ± 0.14 | 0.22 ± 0.09 | 0.16 ± 0.12 | 0.25 ± 0.09 | 50.8 / 47.5 / 51.3 / 70.6 |
| knee | 26.6 ± 1.5 | 105.2 ± 27.1 | 0.12 ± 0.19 | 0.11 ± 0.09 | 0.13 ± 0.16 | 0.10 ± 0.09 | 26.8 / 27.4 / 28.6 / 63.7 |
| ankle | 34.2 ± 1.1 | 103.1 ± 25.7 | 0.11 ± 0.15 | 0.10 ± 0.10 | 0.10 ± 0.16 | 0.10 ± 0.10 | 32.6 / 32.9 / 34.1 / 61.8 |
| foot | 38.5 ± 1.4 | 104.1 ± 25.7 | 0.13 ± 0.15 | 0.11 ± 0.09 | 0.10 ± 0.14 | 0.11 ± 0.10 | 30.0 / 30.5 / 31.8 / 60.0 |
| all | 36.9 ± 1.4 | 108.1 ± 24.8 | 0.13 ± 0.15 | 0.13 ± 0.06 | 0.11 ± 0.10 | 0.14 ± 0.06 | 33.2 / 33.0 / 34.7 / 64.2 |

### Fused landmark error by rule (mm, RMS over frames, mean over landmarks)

| Tasks | Offset removed | mean | invvar | median | wmedian |
|---|---|---|---|---|---|
| all tasks | raw | 51.5 ± 5.4 | 51.2 ± 6.3 | 52.2 ± 5.1 | 88.8 ± 19.7 |
| all tasks | trial | 33.2 ± 2.9 | 33.0 ± 3.1 | 34.7 ± 3.0 | 64.2 ± 13.6 |
| robot tasks | raw | 49.2 ± 6.6 | 49.4 ± 7.4 | 50.6 ± 6.9 | 83.1 ± 20.9 |
| robot tasks | trial | 29.5 ± 4.1 | 29.6 ± 3.9 | 31.3 ± 4.4 | 57.7 ± 12.2 |
| other tasks | raw | 52.7 ± 5.6 | 52.2 ± 6.3 | 53.1 ± 5.2 | 91.7 ± 20.6 |
| other tasks | trial | 35.1 ± 2.9 | 34.7 ± 3.2 | 36.4 ± 2.9 | 67.5 ± 15.2 |

### Calibration: view error by sigma bin (offset removed per trial; frames pooled)

| Sigma bin (mm) | Share of samples, all | Mean error (mm), all | RMS, all | Share, robot | Mean error, robot | RMS, robot |
|---|---|---|---|---|---|---|
| 0 to 20 | 2.0 % | 94.6 | 102.1 | 2.2 % | 94.5 | 100.9 |
| 20 to 25 | 13.9 % | 92.3 | 102.2 | 13.3 % | 87.9 | 96.4 |
| 25 to 30 | 20.9 % | 97.0 | 107.9 | 21.0 % | 92.8 | 102.6 |
| 30 to 35 | 19.3 % | 97.1 | 108.8 | 17.1 % | 93.9 | 104.7 |
| 35 to 40 | 15.0 % | 99.4 | 111.0 | 14.5 % | 92.8 | 103.2 |
| 40 to 50 | 15.0 % | 107.4 | 119.9 | 15.1 % | 100.5 | 111.7 |
| 50 to 60 | 6.7 % | 114.5 | 128.6 | 7.4 % | 108.5 | 121.4 |
| 60 to 80 | 5.3 % | 117.5 | 132.3 | 6.6 % | 112.1 | 124.9 |
| 80 to 120 | 1.8 % | 130.2 | 147.4 | 2.4 % | 121.4 | 135.5 |
| 120 to  | 0.3 % | 178.2 | 219.5 | 0.3 % | 155.1 | 194.2 |

Reading:

- Sigma carries little information on a view's error: rank correlation 0.13 over time and 0.13 across views (0.08 and 0.11 on the robot tasks). The view sigma ranks best is the best 29 % of the time against 25 % by chance.
- A view's error (108 mm RMS) is mostly along its line of sight (96 mm against 46 mm across), about 3 times sigma (37 mm). Sigma does not track the depth error (0.11) and tracks the lateral error less (0.06; 0.00 across views on the robot tasks).
- Sigma is a little more informative at the wrists and elbows (0.22 and 0.17 across views) and there inverse variance lowers the fused landmark error (wrist 47.5 against 50.8 mm); elsewhere it does not. Over all landmarks: 33.0 against 33.2 mm.
- The calibration table is monotone but flat: error grows from 92 to 130 mm while sigma grows from 20 to 120 mm. Only the 0.3 % of samples above 120 mm are clearly worse (178 mm).
- With each view's own mean removed, the correlation over time vanishes (−0.01; −0.14 on robot tasks): what sigma tracks over a trial is mostly which view and landmark, not the moment.
- Fused landmark error: `median` is 1.5 mm worse than `mean`; `wmedian` is about twice as bad (64 against 33 mm).

## 3. Robustness to lost views and outliers

`scripts/python/paper/study_fusion_robustness.py`: fusion, 5 Hz filter and moving-horizon IK as shipped, scored like the other studies (causal error). The clean `mean` replay reproduces the existing `fusion_mean` run byte for byte (checked on 1012 RobotWelding); the 18-participant means are 8.70° (`mean`) and 8.80° (`invvar`), as in the findings.

- **mask**: each view is dropped independently for stretches of frames (mean 1 s) 10, 25 or 50 % of the time. Frames with all four views dropped are lost (10 per trial at 25 %, about 150 at 50 %).
- **outlier**: in one view per trial, one limb (arm or leg landmarks) is displaced by 0.1, 0.3 or 1 m in a random direction for stretches of mean 1 s, 20 % of the time. Sigma is left as recorded, so the outlier is not announced; this cannot favour `invvar`.
- A trial's perturbation is seeded by trial and condition: every rule sees the same one.
- Levels were chosen on the validation participants (folds 0 and 1) by a rule fixed beforehand: the mildest level costing `mean` at least 0.5°, else the strongest. That gave 50 % masking (+0.50°, the strongest level) and 1 m outliers (+2.9°). The other twelve participants were then run on these two levels only.
- 1872 replays, 0 solver failures.

Causal error (deg), mean ± SD over participants; in brackets the mean paired difference to `mean` and its signed-rank p.

### Validation participants (6): 1012 2198 4509 1118 2307 4612

**all tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 8.90 ± 1.07 | 9.09 ± 1.01 (+0.19, p = 0.562) | 8.99 ± 1.02 (+0.09, p = 0.562) | 14.93 ± 3.43 (+6.03, p = 0.031) |
| mask10 | 8.98 ± 1.07 | 9.16 ± 1.00 (+0.18, p = 0.562) | 9.32 ± 1.02 (+0.34, p = 0.031) | 14.10 ± 3.20 (+5.12, p = 0.031) |
| mask25 | 9.19 ± 1.03 | 9.36 ± 0.98 (+0.17, p = 0.688) | 9.58 ± 0.92 (+0.38, p = 0.031) | 13.35 ± 3.02 (+4.16, p = 0.031) |
| mask50 | 9.40 ± 1.06 | 9.52 ± 1.02 (+0.12, p = 0.438) | 9.59 ± 1.02 (+0.19, p = 0.062) | 12.66 ± 1.91 (+3.26, p = 0.031) |
| out10 | 8.97 ± 1.08 | 9.16 ± 1.03 (+0.19, p = 0.562) | 9.02 ± 1.02 (+0.05, p = 0.688) | 14.94 ± 3.35 (+5.97, p = 0.031) |
| out30 | 9.26 ± 0.99 | 9.45 ± 0.93 (+0.19, p = 0.438) | 9.11 ± 1.04 (-0.15, p = 0.219) | 14.99 ± 3.41 (+5.73, p = 0.031) |
| out100 | 11.82 ± 1.09 | 12.07 ± 1.51 (+0.25, p = 0.844) | 9.12 ± 1.00 (-2.70, p = 0.031) | 15.02 ± 3.50 (+3.20, p = 0.062) |

**robot tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 8.28 ± 1.17 | 8.49 ± 1.03 (+0.21, p = 0.438) | 8.37 ± 1.35 (+0.09, p = 0.312) | 14.13 ± 4.67 (+5.86, p = 0.031) |
| mask10 | 8.35 ± 1.20 | 8.55 ± 1.05 (+0.20, p = 0.438) | 8.71 ± 1.52 (+0.36, p = 0.156) | 13.93 ± 4.82 (+5.58, p = 0.031) |
| mask25 | 8.57 ± 1.18 | 8.76 ± 1.01 (+0.19, p = 0.438) | 8.96 ± 1.19 (+0.39, p = 0.031) | 13.35 ± 5.29 (+4.78, p = 0.031) |
| mask50 | 8.82 ± 1.07 | 8.94 ± 1.00 (+0.11, p = 0.688) | 9.03 ± 1.21 (+0.20, p = 0.062) | 12.83 ± 4.22 (+4.01, p = 0.031) |
| out10 | 8.36 ± 1.20 | 8.58 ± 1.07 (+0.22, p = 0.219) | 8.40 ± 1.35 (+0.04, p = 0.688) | 14.16 ± 4.66 (+5.80, p = 0.031) |
| out30 | 8.64 ± 1.10 | 8.85 ± 0.98 (+0.21, p = 0.219) | 8.44 ± 1.32 (-0.20, p = 0.248) | 14.20 ± 4.61 (+5.56, p = 0.031) |
| out100 | 11.48 ± 1.42 | 11.77 ± 1.66 (+0.29, p = 0.528) | 8.50 ± 1.31 (-2.98, p = 0.031) | 14.24 ± 4.66 (+2.76, p = 0.312) |

**other tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 9.21 ± 1.12 | 9.40 ± 1.10 (+0.19, p = 0.562) | 9.29 ± 0.98 (+0.08, p = 0.562) | 15.33 ± 3.16 (+6.12, p = 0.031) |
| mask10 | 9.30 ± 1.11 | 9.46 ± 1.10 (+0.17, p = 0.688) | 9.62 ± 0.92 (+0.33, p = 0.031) | 14.18 ± 2.74 (+4.89, p = 0.031) |
| mask25 | 9.50 ± 1.05 | 9.66 ± 1.06 (+0.16, p = 0.688) | 9.89 ± 0.95 (+0.38, p = 0.031) | 13.35 ± 2.19 (+3.85, p = 0.031) |
| mask50 | 9.69 ± 1.17 | 9.81 ± 1.13 (+0.12, p = 0.438) | 9.87 ± 1.09 (+0.18, p = 0.062) | 12.58 ± 1.41 (+2.89, p = 0.031) |
| out10 | 9.27 ± 1.12 | 9.45 ± 1.11 (+0.18, p = 0.688) | 9.33 ± 0.97 (+0.06, p = 0.688) | 15.33 ± 3.06 (+6.06, p = 0.031) |
| out30 | 9.57 ± 1.05 | 9.75 ± 1.03 (+0.18, p = 0.562) | 9.44 ± 1.01 (-0.13, p = 0.219) | 15.39 ± 3.16 (+5.82, p = 0.031) |
| out100 | 11.99 ± 2.00 | 12.22 ± 2.42 (+0.23, p = 0.562) | 9.42 ± 0.98 (-2.57, p = 0.031) | 15.41 ± 3.27 (+3.42, p = 0.062) |

| Condition | Dropped frames per trial | Solver failures | Marker error (mm) |
|---|---|---|---|
| clean | 0.0 | 0 | 56.4 |
| mask10 | 0.0 | 0 | 58.7 |
| mask25 | 10.0 | 0 | 63.9 |
| mask50 | 145.0 | 0 | 75.1 |
| out10 | 0.0 | 0 | 56.6 |
| out30 | 0.0 | 0 | 57.6 |
| out100 | 0.0 | 0 | 63.0 |

Levels kept for the other participants: mask50, out100 (mildest costing `mean` at least 0.5 deg, else the strongest).

### Held-out participants (12): 1508 1602 1847 2112 3361 4162 4216 4279 4665 4687 4801 4827

**all tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 8.60 ± 0.73 | 8.66 ± 0.80 (+0.06, p = 0.677) | 8.69 ± 0.78 (+0.09, p = 0.027) | 17.77 ± 3.48 (+9.17, p = 0.000) |
| mask50 | 9.21 ± 0.75 | 9.27 ± 0.80 (+0.06, p = 0.470) | 9.39 ± 0.77 (+0.18, p = 0.001) | 13.94 ± 2.44 (+4.73, p = 0.000) |
| out100 | 11.64 ± 0.88 | 11.63 ± 0.84 (-0.01, p = 0.970) | 8.90 ± 0.80 (-2.75, p = 0.000) | 17.84 ± 3.46 (+6.20, p = 0.000) |

**robot tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 7.69 ± 0.97 | 7.81 ± 1.14 (+0.12, p = 0.733) | 7.76 ± 1.05 (+0.07, p = 0.266) | 15.22 ± 1.79 (+7.53, p = 0.000) |
| mask50 | 8.32 ± 0.90 | 8.42 ± 1.05 (+0.10, p = 0.301) | 8.56 ± 0.92 (+0.24, p = 0.000) | 12.04 ± 1.96 (+3.72, p = 0.000) |
| out100 | 11.35 ± 1.59 | 11.29 ± 1.56 (-0.06, p = 0.970) | 8.01 ± 1.07 (-3.34, p = 0.000) | 15.33 ± 1.81 (+3.98, p = 0.000) |

**other tasks**

| Condition | mean | invvar | median | wmedian |
|---|---|---|---|---|
| clean | 9.06 ± 0.71 | 9.09 ± 0.71 (+0.04, p = 0.677) | 9.15 ± 0.75 (+0.09, p = 0.077) | 19.04 ± 4.74 (+9.99, p = 0.000) |
| mask50 | 9.65 ± 0.77 | 9.70 ± 0.76 (+0.05, p = 0.677) | 9.80 ± 0.78 (+0.15, p = 0.003) | 14.89 ± 3.10 (+5.24, p = 0.000) |
| out100 | 11.79 ± 0.96 | 11.80 ± 0.97 (+0.01, p = 0.622) | 9.34 ± 0.74 (-2.45, p = 0.000) | 19.10 ± 4.71 (+7.31, p = 0.000) |

| Condition | Dropped frames per trial | Solver failures | Marker error (mm) |
|---|---|---|---|
| clean | 0.0 | 0 | 54.7 |
| mask50 | 153.7 | 0 | 82.2 |
| out100 | 0.0 | 0 | 61.5 |

Reading (held-out participants):

- **Clean**: `invvar` +0.06° (p = 0.68), `median` +0.09° (p = 0.03): no rule beats the mean. `wmedian` is unusable (17.8°).
- **Lost views**: all rules degrade by about 0.6° at 50 % masking; `median` is 0.18° worse than `mean` (p = 0.001), because with three or two views left it averages fewer of them. Weighting by sigma does not help (+0.06°).
- **Outliers of 1 m**: `mean` and `invvar` lose 3.0° (11.6°); `median` loses 0.2° (8.90°), −2.75° against `mean` (p < 0.001); −3.34° on the robot tasks. On validation, at 0.3 m the gain is −0.15° (not significant) and at 0.1 m nothing: the median pays off only for gross errors.
- Robot tasks behave like the others in every condition.

## 4. Conclusions

- NLF's sigma is weakly informative about a view's 3D error and does not justify weighting: confirmed at the landmark level (33.0 against 33.2 mm) and at the joint level, clean or perturbed. The mean stays the default.
- The unweighted median costs 0.09° on clean data and 0.2° when views are lost, and protects against a grossly wrong view (metre-scale). Such errors were injected, not observed: this study does not say how often NLF produces them. It is worth offering as an option, not as the default.
- The sigma-weighted median, the rule the ticket asked for, should not be used: it picks one view per axis and doubles the landmark error.

Not done: perturbations announced by sigma (a real occlusion may raise it), outliers in more than one view, medians with fewer than four cameras on clean data, statistics corrected for multiplicity (p values above are unadjusted signed-rank tests on participant means; with 6 validation participants the smallest possible p is 0.031).
