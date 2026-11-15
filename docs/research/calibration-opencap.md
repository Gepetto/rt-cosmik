# Human-pose calibration on OpenCap

## Extrinsics against the checkerboard (walking session, relative to Cam0)

| Camera | Subjects | Rotation (deg) | Position (mm) | Baseline (m) | Length (%) |
|---|---|---|---|---|---|
| Cam1 | 9 | 1.96 (0.86 to 4.20) | 106 (44 to 256) | 1.51 | 4.5 |
| Cam2 | 9 | 2.90 (0.50 to 5.79) | 305 (54 to 1068) | 4.17 | 6.4 |
| Cam3 | 9 | 4.00 (1.01 to 7.30) | 344 (132 to 617) | 5.31 | 5.7 |
| Cam4 | 9 | 3.85 (1.37 to 7.38) | 279 (97 to 599) | 6.04 | 3.8 |

## Recordings refused

| Activity | Trials | Refused | Travel (m) |
|---|---|---|---|
| drop_jump | 54 | 35 | 0.38 to 0.57 |
| sit_to_stand | 18 | 18 | 0.10 to 0.18 |
| squat | 18 | 18 | 0.09 to 0.13 |
| walking | 54 | 34 | 0.27 to 0.58 |

## Walking trials under both calibrations (mean over subjects)

| Arm | Subjects | Trials | Joint angles, board (deg) | human (deg) | Markers, board (mm) | human (mm) | human, frame refitted (mm) |
|---|---|---|---|---|---|---|---|
| nlf_2cam | 9 | 54 | 7.95 | 7.98 | 64.6 | 133.7 | 102.8 |
| nlf_3cam | 9 | 54 | 8.01 | 8.04 | 67.0 | 143.9 | 109.8 |
| nlf_5cam | 9 | 54 | 7.45 | 7.47 | 53.5 | 175.3 | 61.8 |

Markers are compared in the mocap frame through one rigid transform per session, fitted on the five-camera arm: under the checkerboard calibration, or refitted under the human-pose calibration (last column).
