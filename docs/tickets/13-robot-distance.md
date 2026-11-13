# 13: Robot-distance statistics

**What to build:** Signed hand-to-end-effector distance errors, 95th and 99th percentiles, and missed close approaches at stated thresholds, computed with causal error, plus the rule for keeping or dropping the application claim.

**Blocked by:** 04

**Status:** code done; rule limits to be confirmed by the author

- [x] Signed errors and percentiles per arm
- [x] Missed close approaches counted at stated thresholds including 105 mm
- [x] Keep-or-drop rule written before final numbers (limits are a proposal)

## Decision rule

Written before any out-of-fold distance number was computed; the limits are a proposal for the author to confirm.

The rule was written first in the working session of 2026-11-13, but rule and result landed in the same commit, so the history does not show the order.

**Quantities.** Distance from each hand centre to the origin of the Panda's hand frame, RobotPolishing and RobotWelding, 18 participants. Error = estimated distance minus reference distance, so a positive error says the hand is farther than it is (the unsafe direction). Causal: estimate and reference at the same video frame, robot in its state at that frame, only the per-trial clock offset removed. Frames pooled over the 36 trials (33 scored, 3 without robot states: 1602 RobotWelding, 1847 RobotPolishing, 2307 RobotWelding), each hand judged separately. Thresholds: 105 mm (the closest approach the submitted paper reports, which R2-10 calls the minimum separation), 150, 200 and 300 mm (up to about the mean separation). A missed frame has the reference below the threshold and the estimate at or above it; the missed rate is missed frames over reference frames below the threshold.

**Rule.** The application is kept, worded as distance estimation and never as safety monitoring, only if the arm the paper presents satisfies all three, causally, for both hands:

1. 99th percentile of the absolute error below 105 mm: in 99 % of frames the error is smaller than the smallest separation it has to resolve.
2. 95th percentile of the signed error below 52.5 mm: the margin that covers 95 % of the overestimates is at most half that separation.
3. Missed rate below 5 % at 105 mm. A threshold with fewer than 100 reference frames below it (2.5 s) is not evaluable; the criterion then applies at the smallest stated threshold that is, and the paper says so. If no stated threshold is evaluable, the criterion is not met.

If any fails, the application claim is dropped: the distance error statistics may stay as a characterisation of position accuracy, with no mention of proximity or collaboration use. False alarms and the aligned figures are reported but do not enter the rule.

**Clarifications added after the numbers (2026-11-13); the limits are unchanged.**

- A reference frame below the threshold with no estimate (dropped frame) is a missed frame, and a threshold is evaluable on its reference frames alone, as the definitions above say; the first code left such frames out of both counts. No frame of the 33 trials is affected.
- Events are counted, never judged by the rule: an approach is a spell of the reference below the threshold lasting at least 5 frames (125 ms at 40 Hz, `hand_distance.MIN_EVENT_FRAMES`), and gaps shorter than that, a missing reference frame included, do not split it. Missed and false-alarm spells are counted the same way. An approach is detected at the first frame within it where the estimate is below the threshold; the delay is counted from the start of the approach.

## Result (computed after the rule was written)

`scripts/python/paper/hand_distance.py` on the out-of-fold runs, outputs in `/root/workspace/oof/t13/v2/`. 33 trials, not 36: COMFI's robot states are empty for 1602/RobotWelding, 1847/RobotPolishing and 2307/RobotWelding (listed in `skipped.csv`). Causal, pooled over frames, mm:

| arm | hand | bias | SD | abs p95 | abs p99 | signed p95 | missed at 105 mm | rule |
|---|---|---|---|---|---|---|---|---|
| nlf_0-2-4-6 | left | -2.5 | 40.4 | 81.2 | 112.7 | 61.3 | 58.5 % | drop |
| nlf_0-2-4-6 | right | +7.9 | 33.0 | 73.9 | 113.3 | 60.9 | 58.2 % | drop |
| fusion_mean | left | -10.8 | 39.6 | 85.3 | 136.3 | 48.8 | 42.8 % | drop |
| fusion_mean | right | +7.6 | 37.8 | 85.4 | 124.9 | 73.3 | 58.0 % | drop |

Both arms fail all three criteria on at least one hand, so the rule drops the application claim. The reference hand centre rests 63 to 105 mm from the hand frame while holding the tool, so 105 mm is inside the working range rather than its minimum, and a 30 to 40 mm error SD cannot resolve it; the misses fall below 9 % at 200 mm and below 0.3 % at 300 mm.

Approaches (causal; never detected / detected late / all) and the delay of the late ones, median and 95th percentile in ms:

| arm | hand | 105 mm | delay | 150 mm | delay | 200 mm | delay | 300 mm | delay |
|---|---|---|---|---|---|---|---|---|---|
| nlf_0-2-4-6 | left | 12 / 13 / 37 | 2375, 13180 | 13 / 33 / 77 | 200, 1135 | 1 / 25 / 88 | 175, 1075 | 0 / 34 / 49 | 75, 175 |
| nlf_0-2-4-6 | right | 12 / 15 / 37 | 350, 21417 | 20 / 61 / 115 | 225, 1075 | 2 / 47 / 69 | 100, 2185 | 0 / 33 / 34 | 100, 135 |
| fusion_mean | left | 10 / 14 / 37 | 1025, 10131 | 5 / 21 / 77 | 200, 2900 | 1 / 26 / 88 | 125, 1031 | 0 / 31 / 49 | 50, 100 |
| fusion_mean | right | 9 / 13 / 37 | 325, 2490 | 22 / 54 / 115 | 275, 1560 | 4 / 47 / 69 | 100, 5397 | 0 / 34 / 34 | 75, 159 |

At 105 mm about a third of the approaches are never detected and the late ones are late by seconds at the 95th percentile. At 300 mm every approach is detected, most of them a few frames late (median 2 to 4 frames).
