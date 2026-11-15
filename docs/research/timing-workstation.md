# Sustained timing

Machine: NVIDIA RTX 4500 Ada Generation, 32 cores, load average 1.5 at start, 2026-11-15 15:38.

Rates are processing rates on recorded video, frames released at the camera rate; a live system is capped by its cameras. The first 40 frames of each trial are left out. Acquisition and synchronisation delays are not in these figures.

## Pipeline stages, COMFI

### NLF backend `torchscript`, detection `per_view`

6 trials, 14991 frames, cameras [0, 2, 4, 6], 40 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.03 | 0.04 | 0.06 | 0.2 |
| upload | 1.70 | 1.81 | 3.00 | 4.2 |
| detection | 7.25 | 9.11 | 11.02 | 18.7 |
| crop | 0.00 | 0.00 | 0.00 | 0.0 |
| nlf | 8.59 | 10.22 | 12.85 | 20.0 |
| fusion | 0.14 | 0.16 | 0.24 | 0.6 |
| filter | 0.03 | 0.03 | 0.05 | 0.3 |
| ik | 3.49 | 4.34 | 5.64 | 9.8 |
| output | 0.16 | 0.25 | 0.31 | 6.3 |
| processing, decode excluded | 22.70 | 25.94 | 30.92 | 50.8 |
| processing and decode | 22.72 | 25.97 | 30.95 | 50.8 |

- Processing rate: 44 frames/s (44 with video decoding).
- Frames not finished when the next one arrives (25.0 ms): 2607 of 14991 (17.39 %); with video decoding 2678 (17.86 %).
- Delay from a frame's arrival to its joint angles: median 22.9 ms, 95th 31.0, 99th 47.1, max 172.9.
- Latency of a motion event, as waiting for the next frame (0 to 25.0 ms), processing and filter lag (10 ms): median 45 ms, at most 82 ms for 99 % of frames.

### NLF backend `tensorrt`, detection `per_view`

6 trials, 14991 frames, cameras [0, 2, 4, 6], 40 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.03 | 0.03 | 0.04 | 0.2 |
| upload | 1.69 | 2.38 | 2.72 | 4.1 |
| detection | 6.65 | 7.72 | 9.14 | 13.9 |
| crop | 2.29 | 3.06 | 4.13 | 5.9 |
| nlf | 1.97 | 2.16 | 2.49 | 3.6 |
| fusion | 0.14 | 0.16 | 0.25 | 0.5 |
| filter | 0.03 | 0.03 | 0.05 | 0.2 |
| ik | 3.34 | 3.97 | 5.83 | 10.0 |
| output | 0.16 | 0.18 | 0.25 | 13.6 |
| processing, decode excluded | 17.49 | 20.05 | 24.82 | 43.4 |
| processing and decode | 17.52 | 20.08 | 24.85 | 43.5 |

- Processing rate: 57 frames/s (57 with video decoding).
- Frames not finished when the next one arrives (25.0 ms): 161 of 14991 (1.07 %); with video decoding 166 (1.11 %).
- Delay from a frame's arrival to its joint angles: median 17.5 ms, 95th 20.2, 99th 25.2, max 44.7.
- Latency of a motion event, as waiting for the next frame (0 to 25.0 ms), processing and filter lag (10 ms): median 40 ms, at most 60 ms for 99 % of frames.

### NLF backend `tensorrt`, detection `tiled`

6 trials, 14991 frames, cameras [0, 2, 4, 6], 40 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.03 | 0.04 | 0.06 | 0.3 |
| upload | 1.75 | 2.35 | 3.37 | 6.3 |
| detection | 3.21 | 3.75 | 4.62 | 7.8 |
| crop | 2.33 | 3.24 | 4.01 | 6.5 |
| nlf | 1.95 | 2.18 | 2.45 | 6.1 |
| fusion | 0.14 | 0.21 | 0.25 | 0.5 |
| filter | 0.03 | 0.04 | 0.05 | 0.1 |
| ik | 3.50 | 4.59 | 6.52 | 10.3 |
| output | 0.16 | 0.20 | 0.32 | 13.7 |
| processing, decode excluded | 14.20 | 16.76 | 20.69 | 37.5 |
| processing and decode | 14.23 | 16.79 | 20.72 | 37.5 |

- Processing rate: 70 frames/s (70 with video decoding).
- Frames not finished when the next one arrives (25.0 ms): 38 of 14991 (0.25 %); with video decoding 39 (0.26 %).
- Delay from a frame's arrival to its joint angles: median 14.2 ms, 95th 16.8, 99th 20.8, max 38.5.
- Latency of a motion event, as waiting for the next frame (0 to 25.0 ms), processing and filter lag (10 ms): median 37 ms, at most 56 ms for 99 % of frames.

## Pipeline stages, OpenCap LabValidation

### NLF backend `torchscript`, detection `per_view`

16 trials, 4293 frames, cameras [0, 1, 3, 4], 60 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.03 | 0.03 | 0.06 | 1.1 |
| upload | 1.70 | 1.94 | 2.68 | 4.3 |
| detection | 7.62 | 9.89 | 11.97 | 15.8 |
| crop | 0.00 | 0.00 | 0.00 | 0.0 |
| nlf | 8.63 | 10.08 | 12.17 | 19.0 |
| fusion | 0.14 | 0.15 | 0.23 | 0.4 |
| filter | 0.03 | 0.03 | 0.04 | 0.1 |
| ik | 3.25 | 3.98 | 5.00 | 8.7 |
| output | 0.15 | 0.18 | 0.29 | 1.5 |
| processing, decode excluded | 22.89 | 26.22 | 29.31 | 38.9 |
| processing and decode | 22.91 | 26.24 | 29.34 | 38.9 |

- Processing rate: 44 frames/s (44 with video decoding).
- Frames not finished when the next one arrives (16.7 ms): 4293 of 4293 (100.00 %); with video decoding 4293 (100.00 %).
- Delay from a frame's arrival to its joint angles: median 2039.0 ms, 95th 5392.5, 99th 6535.5, max 6791.3.
- Latency of a motion event, as waiting for the next frame (0 to 16.7 ms), processing and filter lag (7 ms): median 2054 ms, at most 6559 ms for 99 % of frames.

### NLF backend `tensorrt`, detection `per_view`

16 trials, 4293 frames, cameras [0, 1, 3, 4], 60 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.02 | 0.03 | 0.05 | 0.2 |
| upload | 1.70 | 2.04 | 2.80 | 3.7 |
| detection | 7.57 | 9.72 | 11.68 | 15.3 |
| crop | 2.30 | 2.79 | 3.72 | 6.1 |
| nlf | 1.97 | 2.13 | 2.42 | 3.0 |
| fusion | 0.13 | 0.15 | 0.24 | 0.5 |
| filter | 0.03 | 0.03 | 0.05 | 0.1 |
| ik | 3.24 | 4.02 | 5.23 | 9.8 |
| output | 0.16 | 0.18 | 0.29 | 10.7 |
| processing, decode excluded | 18.42 | 21.69 | 25.00 | 30.8 |
| processing and decode | 18.44 | 21.72 | 25.02 | 30.8 |

- Processing rate: 54 frames/s (54 with video decoding).
- Frames not finished when the next one arrives (16.7 ms): 4293 of 4293 (100.00 %); with video decoding 4293 (100.00 %).
- Delay from a frame's arrival to its joint angles: median 687.4 ms, 95th 1732.6, 99th 2126.7, max 2183.8.
- Latency of a motion event, as waiting for the next frame (0 to 16.7 ms), processing and filter lag (7 ms): median 702 ms, at most 2150 ms for 99 % of frames.

### NLF backend `tensorrt`, detection `tiled`

16 trials, 4293 frames, cameras [0, 1, 3, 4], 60 Hz.

| Stage (ms) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| decode | 0.03 | 0.04 | 0.06 | 0.2 |
| upload | 2.34 | 3.08 | 4.31 | 7.1 |
| detection | 3.29 | 4.31 | 5.23 | 8.0 |
| crop | 2.28 | 3.19 | 3.97 | 6.0 |
| nlf | 1.94 | 2.10 | 2.44 | 3.5 |
| fusion | 0.14 | 0.16 | 0.23 | 0.4 |
| filter | 0.03 | 0.03 | 0.05 | 0.2 |
| ik | 3.42 | 4.32 | 6.07 | 9.6 |
| output | 0.16 | 0.22 | 0.31 | 6.0 |
| processing, decode excluded | 14.82 | 17.58 | 21.89 | 27.9 |
| processing and decode | 14.85 | 17.61 | 21.95 | 28.0 |

- Processing rate: 67 frames/s (67 with video decoding).
- Frames not finished when the next one arrives (16.7 ms): 725 of 4293 (16.89 %); with video decoding 741 (17.26 %).
- Delay from a frame's arrival to its joint angles: median 14.9 ms, 95th 20.9, 99th 27.1, max 36.5.
- Latency of a motion event, as waiting for the next frame (0 to 16.7 ms), processing and filter lag (7 ms): median 30 ms, at most 50 ms for 99 % of frames.

## Inverse kinematics (COMFI)

| IK (ms per frame) | Median | 95th | 99th | Max |
|---|---|---|---|---|
| Moving-horizon IK | 2.80 | 3.59 | 4.36 | 8.6 |
| Per-frame IK | 1.54 | 1.81 | 2.69 | 4.9 |

The moving-horizon IK costs +1.26 ms per frame more than the per-frame IK (medians).

## Camera timestamps of the COMFI recordings

108 trials, 265037 frames per camera.

| Camera | Mean interval (ms) | Frames missing against the nominal rate | Intervals over 1.5 periods | Longest interval (ms) | Offset to camera 0 (ms, range over trials) |
|---|---|---|---|---|---|
| 0 | 25.07 | 770 (0.29 %) | 5888 (2.22 %) | 80 | 0.0 to 0.0 |
| 2 | 25.07 | 769 (0.29 %) | 6669 (2.52 %) | 83 | -17.1 to 17.0 |
| 4 | 25.07 | 764 (0.29 %) | 5705 (2.15 %) | 83 | -16.8 to 16.9 |
| 6 | 25.08 | 789 (0.30 %) | 3750 (1.41 %) | 80 | -17.0 to 17.2 |

Spread of the four cameras' times within a frame: median over trials of the per-trial median 17.1 ms, of the 95th percentile 23.9 ms; largest in any frame 69 ms.
