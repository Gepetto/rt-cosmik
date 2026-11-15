# Human-pose calibration on OpenCap

Extrinsics of each walking trial from the subject's other walking clips, against the dataset's checkerboard calibration; then the walking trials replayed under them. The travel rule is not applied to these calibrations (the clips are short); the refusal table shows what it would refuse, trial by trial. Produced by `scripts/python/paper/opencap_calibration.py`; discussion in `resubmission-findings.md`, section 0f.

## Extrinsics against the checkerboard (walking session, relative to Cam0)

| Camera | Subjects | Rotation (deg) | Position (mm) | Baseline (m) | Length, absolute error (%) |
|---|---|---|---|---|---|
| Cam1 | 9 | 1.96 (0.86 to 4.20) | 106 (44 to 256) | 1.51 | 4.5 |
| Cam2 | 9 | 2.90 (0.50 to 5.79) | 305 (54 to 1068) | 4.17 | 6.4 |
| Cam3 | 9 | 4.00 (1.01 to 7.30) | 344 (132 to 617) | 5.31 | 5.7 |
| Cam4 | 9 | 3.85 (1.37 to 7.38) | 279 (97 to 599) | 6.04 | 3.8 |

## Recordings refused

| Activity | Trials | Refused | Travel (m) |
|---|---|---|---|
| drop_jump | 54 | 54 | 0.35 to 0.50 |
| sit_to_stand | 18 | 18 | 0.10 to 0.23 |
| squat | 18 | 18 | 0.08 to 0.14 |
| walking | 54 | 48 | 0.40 to 0.58 |

## Walking trials under both calibrations (mean over subjects)

| Arm | Subjects | Trials | Joint angles, board (deg) | human (deg) | Markers, board (mm) | human (mm) | human, frame refitted (mm) |
|---|---|---|---|---|---|---|---|
| nlf_2cam | 9 | 54 | 7.95 | 7.98 | 64.6 | 133.7 | 102.8 |
| nlf_3cam | 9 | 54 | 8.01 | 8.04 | 67.0 | 143.9 | 109.8 |
| nlf_5cam | 9 | 54 | 7.45 | 7.47 | 53.5 | 175.3 | 61.8 |

Markers are compared in the mocap frame through one rigid transform per session, fitted on the five-camera arm: under the checkerboard calibration, or refitted under the human-pose calibration (last column).
