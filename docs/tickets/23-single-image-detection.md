# 23: Person detection on one tiled image

**What to build:** The detector runs once per frame set whatever the number of views. The views are tiled into one image, the detector runs on it, and each box is mapped back to the pixel frame of the view it lies in. Today the detector receives one image per view, and at four views it is the largest stage (7.5 ms of 18.4 ms with the TensorRT NLF backend).

**Blocked by:** 22

**Status:** ready-for-agent

- [x] Views are tiled into one image (one row for 2, a 2 x 2 grid for 3 and 4) and the detector is called once
- [x] Each box is assigned to the view holding its centre, clipped to that view and returned in that view's pixel coordinates
- [x] One person per view is kept, as today
- [x] The mode is a setting; per-view detection stays available and stays the default until the comparison below is made
- [x] Compared with per-view detection on the same frames: detection rate per view, box difference, and the resulting landmark and joint angle differences
  - Made on four views of one trial only (1012, Lifting, 300 frames); one to three views and other trials are not compared.
- [ ] Per-stage timing, one to four views, on the workstation GPU
- [x] A single view is unchanged

## What was built and measured

Code: `src/rtcosmik/nlf/tiling.py` (layout, tiling on the GPU, boxes back to each view), `NLFEstimator.detect`, `rtcosmik.model_weights.detector_arguments`. Settings: `detection_mode` (`"per_view"` by default, or `"tiled"`) and `yolo_tiled_imgsz` (height, width). The tiles are cut from the images already uploaded for NLF and resized straight to their place in the detector image. The default image is 736 x 1280: each of four views keeps the 640 x 360 it has in per-view detection. `bash scripts/bash/fetch_models.sh` builds its engine when `detection_mode` is `"tiled"`; a square 640 image runs on the single-camera engine. Tiles are rounded to whole pixels and boxes are mapped back with one scale per axis.

Tiled against per-view detection, participant 1012, Lifting, cameras 0-2-4-6, first 300 frames, TensorRT NLF backend (`scripts/python/paper/detection_mode_parity.py`). Per-view detection sees each view at 640 x 360.

| Detector image | Pixels per view | Detection rate per view | Box IoU median (worst view) | IoU min | Landmark median | p99 | max | Joint angle median | max |
|---|---|---|---|---|---|---|---|---|---|
| per view (reference) | 640 x 360 | 0.997 1 1 1 | | | | | | | |
| 640 x 640 | 320 x 180 | 0.997 1 1 1 | 0.917 | 0.70 | 13.0 mm | 77.7 mm | 174.7 mm | 0.39 deg | 11.4 deg |
| 384 x 640 | 320 x 180 | 0.997 1 1 1 | 0.919 | 0.70 | 12.9 mm | 69.6 mm | 172.1 mm | 0.34 deg | 10.0 deg |
| 544 x 960 | 480 x 270 | 0.977 1 0.997 1 | 0.945 | 0.71 | 10.1 mm | 52.6 mm | 141.3 mm | 0.33 deg | 7.2 deg |
| 736 x 1280 | 640 x 360 | 0.997 1 1 1 | 0.975 | 0.75 | 6.7 mm | 43.9 mm | 108.1 mm | 0.18 deg | 4.0 deg |

The TorchScript backend gives the same figures (640 x 640: 13.0 mm, 0.38 deg; 736 x 1280: 6.8 mm, 0.18 deg). Landmark differences are per view, in the camera frame, before fusion. These are differences to per-view detection, not errors: accuracy against mocap is not measured yet, and neither is the timing.
