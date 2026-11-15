# Human-pose calibration on COMFI

## Extrinsics against the dataset calibration (median, quartiles, maximum)

| Recording | Pair | Cameras | Baseline (m) | Rotation (deg) | Position (mm) | Length (%) |
|---|---|---|---|---|---|---|
| CircularWalking | adjacent | 18 | 0.81 | 0.84 (0.62, 1.27; 1.99) | 40.21 (29.40, 57.63; 112.61) | 4.25 (2.74, 6.58; 13.60) |
| CircularWalking | opposite | 36 | 5.34 | 1.58 (0.92, 2.15; 3.62) | 194.74 (127.90, 238.93; 316.92) | 2.85 (1.85, 3.55; 4.46) |
| StraightWalking | adjacent | 18 | 0.81 | 0.66 (0.52, 0.78; 1.06) | 41.99 (27.94, 47.00; 61.30) | 2.90 (1.82, 4.80; 6.35) |
| StraightWalking | opposite | 36 | 5.34 | 1.43 (1.08, 2.05; 2.74) | 180.64 (138.92, 198.91; 304.70) | 3.03 (1.96, 3.49; 5.34) |

## Recordings refused

| Recording | Trials | Refused | Travel (m) |
|---|---|---|---|
| CircularWalking | 18 | 0 | 0.57 to 0.89 |
| Lifting | 18 | 17 | 0.18 to 0.52 |
| Polishing | 18 | 18 | 0.09 to 0.38 |
| RobotPolishing | 18 | 18 | 0.12 to 0.46 |
| RobotWelding | 18 | 17 | 0.12 to 0.53 |
| Screwing | 18 | 18 | 0.10 to 0.36 |
| SideOverhead | 18 | 14 | 0.30 to 0.67 |
| StraightWalking | 18 | 0 | 0.77 to 1.17 |

## Evaluation tasks under each calibration (mean over participants)

| Calibration | Trials | Joint angles (deg) | Difference (deg, mean; largest participant) | Markers (mm) | Worst participant (mm) |
|---|---|---|---|---|---|
| dataset | 108 | 8.16 | +0.000; 0.00 | 50.0 | 63 |
| calib_human | 108 | 8.16 | -0.004; 0.03 | 98.0 | 133 |
| rot_0.5 | 108 | 8.16 | +0.000; 0.01 | 50.5 | 64 |
| rot_1 | 108 | 8.16 | +0.001; 0.01 | 52.6 | 63 |
| rot_2 | 108 | 8.16 | -0.001; 0.02 | 59.0 | 74 |
| rot_5 | 108 | 8.16 | -0.001; 0.02 | 93.6 | 111 |
| shift_10 | 108 | 8.16 | +0.000; 0.00 | 50.3 | 64 |
| shift_20 | 108 | 8.16 | +0.000; 0.00 | 50.5 | 66 |
| shift_50 | 108 | 8.16 | +0.000; 0.00 | 53.9 | 69 |
| shift_100 | 108 | 8.16 | +0.000; 0.00 | 62.9 | 75 |

## Hand-to-robot distance on the robot tasks

| Calibration | Hand | Trials | Bias (mm) | SD (mm) | 95th pct. of abs. error (mm) | Missed at 300 mm (%) | False alarms at 300 mm (%) |
|---|---|---|---|---|---|---|---|
| fusion_mean | left | 33 | -11.3 | 37.6 | 81.2 | 0.06 | 8.61 |
| fusion_mean | right | 33 | +7.2 | 33.4 | 77.1 | 0.07 | 0.30 |
| calib_human | left | 33 | -13.4 | 51.6 | 109.1 | 0.05 | 11.97 |
| calib_human | right | 33 | +24.0 | 47.7 | 124.8 | 0.26 | 0.34 |
| rot_1 | left | 33 | -11.2 | 39.4 | 86.0 | 0.08 | 8.54 |
| rot_1 | right | 33 | +8.8 | 33.6 | 74.7 | 0.07 | 0.27 |
| rot_2 | left | 33 | -9.5 | 41.9 | 92.4 | 0.10 | 8.68 |
| rot_2 | right | 33 | +6.4 | 39.4 | 85.6 | 0.09 | 0.43 |
| rot_5 | left | 33 | +4.9 | 67.3 | 135.7 | 4.55 | 10.72 |
| rot_5 | right | 33 | +32.4 | 63.2 | 129.9 | 0.23 | 0.51 |
| shift_100 | left | 33 | -9.1 | 43.3 | 87.9 | 0.84 | 7.69 |
| shift_100 | right | 33 | +6.6 | 41.4 | 83.0 | 0.07 | 0.41 |
| shift_20 | left | 33 | -11.8 | 38.6 | 82.9 | 0.10 | 8.95 |
| shift_20 | right | 33 | +6.6 | 33.3 | 78.0 | 0.06 | 0.30 |
| shift_50 | left | 33 | -10.3 | 41.9 | 89.0 | 0.23 | 8.05 |
| shift_50 | right | 33 | +6.9 | 36.4 | 78.2 | 0.07 | 0.39 |
