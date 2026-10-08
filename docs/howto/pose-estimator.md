# Use your own pose estimator

The reconstruction does not depend on NLF. Any estimator that returns, for each
camera, the 3D position of RT-COSMIK's 43 landmarks can feed the same fusion,
filter and inverse kinematics: a faster network for live use, or a slower
foundation model for offline processing. The paper evaluated Fast SAM 3D Body
this way.

## What the estimator must return

For each camera and each frame:

- **The landmarks**: a `(43, 3)` array, in metres, in that camera's frame
  (OpenCV axes: x right, y down, z forward), one row per landmark in the order
  of `settings.marker_names`. `None` when the camera saw nobody.
- **Optionally, their uncertainty**: one value per camera and landmark, as a
  `(cameras, 43)` array. Views are weighted by the inverse of its square, so an
  occluded landmark in one view counts less in that view only. Without
  uncertainties, the views are simply averaged.

The landmarks are anatomical points, listed in
[Outputs](../outputs.md#markerscsv). An estimator built on a body model gives
them as vertices of its mesh: NLF is queried at the SMPL-X vertices of
`settings.nlf_indices`. For another model, the paper picked the vertex nearest
to each marker by a small optimization against motion capture.

## The loop

A complete loop, from your estimator's output to joint angles, using
RT-COSMIK's own fusion, filter and solver. `synchronized_images` and
`my_estimator` stand for your acquisition and your estimator.

```python
from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from rtcosmik.config_loader import settings
from rtcosmik.filtering.iir import MarkerFilter
from rtcosmik.nlf.nlf import Views
from rtcosmik.pipeline.solver import HumanSolver
from rtcosmik.triangulation.triangulation import reconstruct_3d

cameras = [0, 2, 4, 6]
calibration = "data/comfi_sample/cam_params/2112"
mtxs, dists, projections, _, _ = load_camera_parameters(calibration, cameras)
world_R, world_T = load_world_transformation(calibration, cameras[0])

marker_filter = MarkerFilter(len(settings.marker_names), settings)
solver = HumanSolver(settings, height=1.77, weight=62.0, gender="f")

for images in synchronized_images():           # one image per camera
    poses3d, sigma = my_estimator(images)      # see "What the estimator must return"
    views = Views(keypoints=[None] * len(cameras), poses3d=poses3d,
                  uncertainties=sigma,
                  valid_cam_ids=[c for c, p in enumerate(poses3d) if p is not None])
    points = reconstruct_3d(views, projections)       # (43, 3), reference camera frame
    if len(points) == 0:
        continue                                      # nobody seen in this frame
    landmarks = marker_filter(points @ world_R.T + world_T)   # world frame, filtered
    q = solver.solve(dict(zip(settings.marker_names, landmarks)))
```

`q` is the model's configuration, laid out as the columns of
`joint_angles.csv` (see [Outputs](../outputs.md)). The first frame calibrates
the model to the person, who should stand still. The solver must have been
generated for the frame rate of your images (`fs`, see
[Inverse kinematics](../inverse-kinematics.md)).

## Estimators that only give 2D points

Keypoint detectors give pixel positions rather than metric 3D points. Pass them
to `triangulate_points` instead of `reconstruct_3d`: it triangulates each
landmark from every camera that saw it, weighted by the uncertainties when they
are given, and returns the same `(43, 3)` array in the reference camera frame.

```python
from rtcosmik.triangulation.triangulation import triangulate_points

points = triangulate_points(keypoints, mtxs, dists, projections, uncertainties=sigma)
```

`keypoints` holds one `(43, 2)` array of pixels per camera, or `None`. It needs
at least two cameras, and degrades when cameras face each other, where 3D
fusion does not.

## Using it in the pipeline

There is no plug-in mechanism yet: `run_pipeline.py` (offline) and
`PipelineProcess` (live) build an `NLFEstimator` and turn its output into
`Views` with `extract_views`. To run the full pipeline with your estimator,
replace those two calls with yours; everything after them stays as it is.

## A different set of landmarks

The 43 landmarks are tied to the model: each is registered on a segment
(`SGTS_MKS_MAPPING` in `rtcosmik.human_model.model_utils`), and the solver is
generated for the landmarks it tracks (`keys_to_track_list`). An estimator with
other landmarks needs both updated, `marker_names` to match, and the solver
regenerated.
