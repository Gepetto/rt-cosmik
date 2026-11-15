# Tables of the paper (COMFI)

Mean (SD) over the 18 participants; errors are causal (no time alignment). Produced by `scripts/python/paper/paper_tables.py`.

## Joint-angle error (deg) and marker error

| Front end | Cameras | Whole body | Without pronation | Lower limbs | Trunk | Upper limbs | Mean abs. bias | SD | Markers (mm) | Lag (frames) |
|---|---|---|---|---|---|---|---|---|---|---|
| RTMPose+LSTM | 4 | 12.34 (1.05) | 10.86 (1.12) | 9.35 (0.83) | 9.78 (1.70) | 16.40 (1.93) | 8.37 (0.82) | 7.77 (0.87) | 75.6 (17.3) | 1.9 |
| NLF-2D | 4 | 9.05 (0.98) | 7.45 (0.83) | 6.13 (0.78) | 7.55 (1.36) | 12.59 (1.86) | 6.09 (1.04) | 5.75 (0.35) | 69.3 (15.5) | 0.5 |
| NLF-2D | 2S | 12.54 (1.28) | 10.48 (1.27) | 8.18 (1.14) | 10.92 (2.01) | 17.58 (2.31) | 7.35 (1.06) | 8.94 (1.12) | 92.8 (26.3) | 0.6 |
| NLF-2D | 2F | 22.11 (8.35) | 20.69 (8.30) | 15.75 (6.75) | 20.37 (9.32) | 29.20 (10.52) | 14.07 (7.06) | 14.18 (3.75) | 101.4 (30.9) | -1.2 |
| NLF-3D | 4 | 8.16 (0.82) | 6.69 (0.70) | 5.68 (0.75) | 6.30 (1.45) | 11.42 (1.50) | 5.80 (0.79) | 4.87 (0.41) | 50.0 (4.8) | 0.4 |
| NLF-3D | 2S | 8.69 (0.88) | 7.35 (0.67) | 6.32 (0.72) | 6.70 (1.49) | 11.90 (1.77) | 5.96 (0.85) | 5.46 (0.45) | 104.2 (23.0) | 0.6 |
| NLF-3D | 2F | 8.38 (0.86) | 6.96 (0.73) | 5.83 (0.82) | 6.43 (1.40) | 11.75 (1.61) | 5.85 (0.79) | 5.13 (0.48) | 57.6 (6.0) | 0.5 |
| NLF-3D | 1 | 9.03 (0.94) | 7.74 (0.66) | 6.61 (0.83) | 7.00 (1.43) | 12.29 (1.85) | 6.05 (0.89) | 5.84 (0.49) | 115.2 (24.8) | 0.5 |

## Per degree of freedom, NLF-3D, four cameras (deg)

| DoF | RMSE | Bias | SD |
|---|---|---|---|
| Cervical_Flexion_Extension | 10.47 (5.35) | -8.74 | 3.88 |
| Cervical_Internal_External_Rotation | 6.89 (1.78) | +1.57 | 4.77 |
| Cervical_Lateral_Bending | 7.48 (2.73) | -2.12 | 5.23 |
| Left_Ankle_Inversion_Eversion | 7.79 (2.81) | -2.74 | 4.04 |
| Left_Ankle_Plantarflexion_Dorsiflexion | 4.94 (1.99) | +2.11 | 2.59 |
| Left_Clavicle_Elevation_Depression | 6.18 (1.62) | +2.58 | 4.09 |
| Left_Elbow_Flexion_Extension | 7.91 (2.25) | +1.16 | 5.59 |
| Left_Elbow_Pronation_Supination | 33.16 (17.86) | -6.66 | 13.87 |
| Left_Hip_Abduction_Adduction | 4.28 (1.41) | +1.87 | 2.73 |
| Left_Hip_Flexion_Extension | 4.39 (1.59) | +1.75 | 2.69 |
| Left_Hip_Internal_External_Rotation | 7.50 (1.71) | -0.86 | 5.19 |
| Left_Knee_Flexion_Extension | 3.26 (0.80) | +0.94 | 2.29 |
| Left_Shoulder_Abduction_Adduction | 7.95 (1.71) | -3.13 | 5.84 |
| Left_Shoulder_Flexion_Extension | 6.23 (1.72) | -1.65 | 4.75 |
| Left_Shoulder_Internal_External_Rotation | 11.08 (3.03) | -5.23 | 7.52 |
| Lumbar_Flexion_Extension | 3.14 (0.70) | -0.29 | 2.13 |
| Lumbar_Lateral_Bending | 3.52 (1.31) | +1.41 | 2.25 |
| Right_Ankle_Inversion_Eversion | 8.02 (2.74) | +4.91 | 4.60 |
| Right_Ankle_Plantarflexion_Dorsiflexion | 5.58 (2.49) | +0.25 | 2.88 |
| Right_Clavicle_Elevation_Depression | 6.44 (2.06) | +3.53 | 3.92 |
| Right_Elbow_Flexion_Extension | 8.73 (1.74) | -5.69 | 5.00 |
| Right_Elbow_Pronation_Supination | 22.75 (5.48) | -10.11 | 13.73 |
| Right_Hip_Abduction_Adduction | 4.30 (1.29) | +1.83 | 2.70 |
| Right_Hip_Flexion_Extension | 5.84 (2.12) | +3.09 | 2.76 |
| Right_Hip_Internal_External_Rotation | 7.52 (2.22) | +2.20 | 4.86 |
| Right_Knee_Flexion_Extension | 4.78 (1.42) | +1.97 | 2.44 |
| Right_Shoulder_Abduction_Adduction | 9.76 (3.09) | -5.94 | 6.29 |
| Right_Shoulder_Flexion_Extension | 5.64 (1.35) | +1.29 | 4.38 |
| Right_Shoulder_Internal_External_Rotation | 11.15 (1.88) | -2.00 | 8.38 |

## Per task, four cameras (deg)

| Front end | Lifting | Polishing | RobotPolishing | RobotWelding | Screwing | SideOverhead |
|---|---|---|---|---|---|---|
| NLF-3D | 7.72 (1.15) | 8.54 (1.00) | 7.42 (1.09) | 7.59 (1.20) | 8.70 (0.84) | 9.00 (1.06) |
| RTMPose+LSTM | 12.74 (1.65) | 12.39 (1.60) | 11.36 (1.47) | 11.42 (2.18) | 11.98 (1.28) | 14.17 (1.41) |

## Posture-only REBA over whole trials

| Front end | Cameras | Angles | MAE (points) | Same risk level (%) | Weighted kappa | High-risk frames rated lower (%) |
|---|---|---|---|---|---|---|
| RTMPose+LSTM | 4 | absolute | 1.05 (0.15) | 61.4 (6.3) | 0.45 (0.10) | 51.3 (17.3) |
| RTMPose+LSTM | 4 | neutral_reference | 1.24 (0.31) | 60.0 (10.0) | 0.49 (0.13) | 44.2 (17.7) |
| RTMPose+LSTM | 4 | neutral_own | 1.00 (0.17) | 65.6 (7.1) | 0.57 (0.08) | 60.1 (22.0) |
| NLF-2D | 4 | absolute | 0.94 (0.19) | 66.0 (7.1) | 0.54 (0.10) | 35.1 (19.9) |
| NLF-2D | 4 | neutral_reference | 0.95 (0.22) | 65.6 (6.6) | 0.56 (0.09) | 46.9 (18.8) |
| NLF-2D | 4 | neutral_own | 0.84 (0.16) | 70.2 (5.9) | 0.64 (0.07) | 47.8 (20.6) |
| NLF-3D | 4 | absolute | 0.91 (0.17) | 62.9 (10.2) | 0.53 (0.12) | 45.6 (19.7) |
| NLF-3D | 4 | neutral_reference | 0.82 (0.18) | 72.4 (7.0) | 0.66 (0.09) | 51.4 (23.4) |
| NLF-3D | 4 | neutral_own | 0.71 (0.14) | 75.1 (6.2) | 0.70 (0.07) | 54.1 (22.8) |
| NLF-3D | 2S | absolute | 0.96 (0.18) | 62.3 (8.7) | 0.52 (0.11) | 44.7 (21.8) |
| NLF-3D | 2S | neutral_reference | 0.80 (0.16) | 72.5 (6.3) | 0.66 (0.08) | 47.7 (20.5) |
| NLF-3D | 2S | neutral_own | 0.74 (0.15) | 74.1 (5.3) | 0.69 (0.05) | 52.8 (19.9) |
| NLF-3D | 2F | absolute | 0.92 (0.17) | 62.7 (9.3) | 0.53 (0.12) | 46.2 (18.3) |
| NLF-3D | 2F | neutral_reference | 0.85 (0.20) | 71.3 (7.5) | 0.64 (0.09) | 52.8 (24.6) |
| NLF-3D | 2F | neutral_own | 0.71 (0.15) | 74.8 (6.5) | 0.70 (0.07) | 53.9 (22.8) |
| NLF-3D | 1 | absolute | 0.95 (0.18) | 62.4 (8.5) | 0.52 (0.11) | 46.8 (20.3) |
| NLF-3D | 1 | neutral_reference | 0.86 (0.18) | 70.7 (6.5) | 0.64 (0.08) | 48.6 (19.6) |
| NLF-3D | 1 | neutral_own | 0.78 (0.16) | 72.4 (6.1) | 0.67 (0.06) | 54.9 (22.2) |

## Posture-only REBA, NLF-3D against RTMPose+LSTM, four cameras (not corrected)

| Quantity (own neutral posture) | NLF-3D | RTMPose+LSTM | p |
|---|---|---|---|
| MAE over whole trials | 0.71 | 1.00 | 7.6e-06 |
| Weighted kappa | 0.70 | 0.57 | 7.6e-06 |
| MAE at the peak posture | 0.77 | 1.19 | 0.0064 |

## Posture-only REBA at the peak posture picked on the estimate (own neutral posture)

| Front end | Cameras | Task | Postures | Reference score | Estimated score | MAE (points) | Same risk level (%) |
|---|---|---|---|---|---|---|---|
| RTMPose+LSTM | 4 | Lifting | 18 | 8.22 (0.81) | 8.94 (1.00) | 0.83 (0.71) | 83.3 |
| RTMPose+LSTM | 4 | Polishing | 18 | 5.28 (1.90) | 6.89 (1.32) | 1.72 (1.67) | 55.6 |
| RTMPose+LSTM | 4 | RobotPolishing | 18 | 4.89 (1.94) | 6.17 (1.72) | 1.72 (1.53) | 50.0 |
| RTMPose+LSTM | 4 | RobotWelding | 18 | 4.67 (1.53) | 5.83 (1.95) | 1.17 (1.38) | 72.2 |
| RTMPose+LSTM | 4 | Screwing | 18 | 5.89 (2.19) | 6.39 (1.65) | 0.83 (0.79) | 72.2 |
| RTMPose+LSTM | 4 | SideOverhead | 18 | 7.17 (1.20) | 7.67 (0.84) | 0.83 (0.79) | 50.0 |
| RTMPose+LSTM | 4 | All tasks | 108 | 6.02 (2.07) | 6.98 (1.78) | 1.19 (1.25) | 63.9 |
| NLF-2D | 4 | Lifting | 18 | 8.22 (0.88) | 9.33 (0.77) | 1.22 (0.65) | 72.2 |
| NLF-2D | 4 | Polishing | 18 | 5.89 (1.37) | 6.56 (1.29) | 1.11 (1.23) | 88.9 |
| NLF-2D | 4 | RobotPolishing | 18 | 5.17 (1.98) | 6.06 (1.51) | 1.56 (1.04) | 61.1 |
| NLF-2D | 4 | RobotWelding | 18 | 5.06 (1.95) | 5.83 (1.89) | 1.22 (1.06) | 61.1 |
| NLF-2D | 4 | Screwing | 18 | 6.11 (1.97) | 6.06 (2.18) | 0.83 (0.92) | 83.3 |
| NLF-2D | 4 | SideOverhead | 18 | 7.17 (1.34) | 7.78 (0.73) | 0.72 (0.89) | 66.7 |
| NLF-2D | 4 | All tasks | 108 | 6.27 (1.95) | 6.94 (1.93) | 1.11 (1.00) | 72.2 |
| NLF-3D | 4 | Lifting | 18 | 8.44 (0.70) | 8.28 (0.89) | 0.28 (0.57) | 83.3 |
| NLF-3D | 4 | Polishing | 18 | 5.61 (1.50) | 5.28 (1.27) | 1.00 (0.97) | 94.4 |
| NLF-3D | 4 | RobotPolishing | 18 | 5.50 (1.86) | 6.11 (1.49) | 0.94 (0.80) | 72.2 |
| NLF-3D | 4 | RobotWelding | 18 | 4.94 (1.73) | 5.39 (1.85) | 1.00 (1.14) | 66.7 |
| NLF-3D | 4 | Screwing | 18 | 5.94 (2.01) | 6.50 (1.69) | 0.78 (0.88) | 66.7 |
| NLF-3D | 4 | SideOverhead | 18 | 7.33 (1.03) | 7.50 (0.86) | 0.61 (0.70) | 55.6 |
| NLF-3D | 4 | All tasks | 108 | 6.30 (1.93) | 6.51 (1.74) | 0.77 (0.88) | 73.1 |
| NLF-3D | 2S | Lifting | 18 | 8.39 (0.78) | 8.33 (0.91) | 0.50 (0.71) | 83.3 |
| NLF-3D | 2S | Polishing | 18 | 5.50 (1.54) | 5.78 (1.40) | 1.28 (1.07) | 83.3 |
| NLF-3D | 2S | RobotPolishing | 18 | 5.17 (1.72) | 6.33 (1.88) | 1.28 (1.02) | 61.1 |
| NLF-3D | 2S | RobotWelding | 18 | 4.56 (1.72) | 5.39 (1.65) | 1.17 (0.71) | 55.6 |
| NLF-3D | 2S | Screwing | 18 | 5.72 (1.99) | 6.56 (1.76) | 0.94 (0.80) | 66.7 |
| NLF-3D | 2S | SideOverhead | 18 | 7.50 (0.99) | 7.72 (0.57) | 0.56 (0.78) | 72.2 |
| NLF-3D | 2S | All tasks | 108 | 6.14 (2.01) | 6.69 (1.75) | 0.95 (0.90) | 70.4 |
| NLF-3D | 2F | Lifting | 18 | 8.50 (0.71) | 8.33 (0.84) | 0.28 (0.46) | 88.9 |
| NLF-3D | 2F | Polishing | 18 | 5.44 (1.50) | 5.56 (1.34) | 1.44 (1.04) | 77.8 |
| NLF-3D | 2F | RobotPolishing | 18 | 5.50 (1.86) | 6.28 (1.60) | 1.22 (0.94) | 72.2 |
| NLF-3D | 2F | RobotWelding | 18 | 5.06 (1.80) | 5.56 (1.65) | 1.06 (1.16) | 72.2 |
| NLF-3D | 2F | Screwing | 18 | 5.67 (1.94) | 6.28 (1.67) | 0.83 (0.79) | 72.2 |
| NLF-3D | 2F | SideOverhead | 18 | 7.44 (1.04) | 7.56 (0.70) | 0.44 (0.62) | 83.3 |
| NLF-3D | 2F | All tasks | 108 | 6.27 (1.97) | 6.59 (1.68) | 0.88 (0.94) | 77.8 |
| NLF-3D | 1 | Lifting | 18 | 8.33 (0.77) | 8.28 (0.83) | 0.39 (0.50) | 88.9 |
| NLF-3D | 1 | Polishing | 18 | 5.67 (1.28) | 5.78 (1.26) | 1.33 (0.91) | 94.4 |
| NLF-3D | 1 | RobotPolishing | 18 | 5.50 (1.79) | 6.50 (1.95) | 1.11 (1.02) | 77.8 |
| NLF-3D | 1 | RobotWelding | 18 | 4.89 (1.57) | 5.33 (1.53) | 0.67 (0.77) | 72.2 |
| NLF-3D | 1 | Screwing | 18 | 5.50 (2.12) | 6.44 (1.92) | 1.06 (1.11) | 66.7 |
| NLF-3D | 1 | SideOverhead | 18 | 7.28 (1.32) | 7.61 (0.61) | 0.78 (0.73) | 77.8 |
| NLF-3D | 1 | All tasks | 108 | 6.19 (1.93) | 6.66 (1.74) | 0.89 (0.90) | 79.6 |

## Hand to end effector on the robot tasks, all trials pooled

| Front end | Cameras | Hand | Trials | Reference distance (mm) | Bias (mm) | SD (mm) | RMSE (mm) | Overstated by, 95th pct. (mm) | 99th pct. (mm) | Missed at 300 mm (%) | False alarms at 300 mm (%) | Missed at 105 mm (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTMPose+LSTM | 4 | left | 33 | 294 | -9.0 | 109.4 | 109.8 | 61 | 78 | 0.28 | 12.13 | 8 |
| RTMPose+LSTM | 4 | right | 33 | 308 | +45.0 | 90.7 | 101.3 | 176 | 226 | 2.56 | 0.22 | 88 |
| NLF-2D | 4 | left | 33 | 294 | -13.5 | 45.3 | 47.3 | 50 | 69 | 0.03 | 11.30 | 20 |
| NLF-2D | 4 | right | 33 | 308 | +31.4 | 39.2 | 50.2 | 115 | 163 | 0.41 | 0.14 | 84 |
| NLF-3D | 4 | left | 33 | 294 | -11.3 | 37.6 | 39.3 | 44 | 71 | 0.06 | 8.61 | 42 |
| NLF-3D | 4 | right | 33 | 308 | +7.2 | 33.4 | 34.2 | 66 | 109 | 0.07 | 0.30 | 58 |
| NLF-3D | 2S | left | 33 | 294 | -2.0 | 64.9 | 64.9 | 99 | 151 | 2.29 | 12.86 | 44 |
| NLF-3D | 2S | right | 33 | 308 | +28.0 | 50.8 | 58.0 | 120 | 209 | 1.47 | 0.41 | 73 |
| NLF-3D | 2F | left | 33 | 294 | -12.8 | 41.5 | 43.5 | 46 | 74 | 0.14 | 9.02 | 34 |
| NLF-3D | 2F | right | 33 | 308 | +16.3 | 41.3 | 44.4 | 102 | 155 | 0.19 | 0.30 | 66 |
| NLF-3D | 1 | left | 33 | 294 | +6.2 | 72.6 | 72.8 | 124 | 188 | 4.61 | 11.56 | 48 |
| NLF-3D | 1 | right | 33 | 308 | +41.6 | 64.9 | 77.1 | 165 | 258 | 4.38 | 0.45 | 73 |

## Runs behind each arm

| Arm | Trials | Participants | Commit | Runs with uncommitted code | Landmark maps | Runs with an in-sample map |
|---|---|---|---|---|---|---|
| RTMPose+LSTM, 4 | 108 | 18 | 209511c | 0 | no fitted map | 0 |
| NLF-2D, 4 | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-2D, 2S | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-2D, 2F | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-3D, 4 | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-3D, 2S | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-3D, 2F | 108 | 18 | 209511c | 0 | 6 folds | 0 |
| NLF-3D, 1 | 108 | 18 | 209511c | 0 | 6 folds | 0 |
