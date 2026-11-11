# Overview

RT-COSMIK turns synchronized camera images into the motion of a biomechanical
model. For every frame, it returns the configuration of a whole-body human
model: the position and orientation of the pelvis in the world, and 36 joint
angles defined as the International Society of Biomechanics recommends. It also
returns the 43 3D landmarks the model was fitted to. Both are produced at the
camera rate, 40 Hz with four 720p webcams on one workstation.

![The RT-COSMIK pipeline: synchronized camera images, 3D landmarks estimated in each view, landmarks fused over a sliding window, and the biomechanical model fitted to them.](assets/pipeline.jpg)

## The pipeline

### 1. Landmarks in each view

A person detector (YOLO, as a TensorRT engine) finds the operator in every image,
in one batch over all cameras, and keeps tracking the same person in each view
from frame to frame. [NLF](https://github.com/isarandi/nlf) (Neural Localizer
Fields) then regresses, for each view, the 3D position of 43 landmarks: points
of the SMPL-X body template placed where anatomical markers sit, on the pelvis,
the spine, the limbs, the hands, the feet and the head. Positions are metric,
in metres, in that camera's own frame, and each comes with an uncertainty. The
landmarks are listed in [Outputs](outputs.md#markerscsv); the template points
they correspond to are `nlf_indices` in `settings.py`.

### 2. Fusion of the views

The landmarks of every view are brought into the frame of the reference camera,
the first of `cameras`, and averaged, each view weighted by the inverse of its
variance:

$$
\mathbf{y}_i = \frac{\sum_{c} w_{c,i}\, \mathbf{R}_c^\top (\mathbf{p}_{c,i} - \mathbf{t}_c)}{\sum_{c} w_{c,i}},
\qquad w_{c,i} = \sigma_{c,i}^{-2}
$$

where $\mathbf{p}_{c,i}$ is landmark $i$ in camera $c$, $\sigma_{c,i}$ its
uncertainty, and $(\mathbf{R}_c, \mathbf{t}_c)$ maps the reference frame to
camera $c$. A landmark a view is unsure of, typically an occluded one, counts
less in that view only. The result is then expressed in the world frame.

Each view's landmarks form a coherent body, so averaging them keeps segment
lengths stable and cancels the depth error a single view has. With one camera,
the fusion returns that camera's estimate. Triangulation of 2D keypoints is
also available, for pose estimators that only give 2D points (see
[Use your own pose estimator](howto/pose-estimator.md)), but it is less
accurate, and ill-conditioned when cameras face each other.

### 3. Filtering

The fused landmarks are low-pass filtered by a causal Butterworth filter, one
frame at a time: fourth order, 10 Hz cutoff by default (`order`, `cutoff_freq`).
A lower cutoff is smoother and lags more.

### 4. Fitting the model to the person, once

On the first frame, the human model of
[example-robot-data](https://github.com/Gepetto/example-robot-data/tree/devel/robots/human_description)
is scaled from the person's height, mass and sex with anthropometric tables,
then its segment lengths and the positions of the landmarks on its segments are
set from the measured landmarks. The person should stand still and fully in
view for this. The model is described in [Outputs](outputs.md#the-human-model).

### 5. Inverse kinematics

Every following frame, the model is fitted to the landmarks of the last `N`
frames at once (10 by default), under joint limits, with a cost that favours
smooth motion: a moving-horizon estimation, solved by acados in a few
milliseconds. The newest pose of the window is the result. The solver is
generated once for everybody; a new person only changes its parameters. See
[Inverse kinematics](inverse-kinematics.md).

## Live and offline

**Live** (`run_pipeline.py --online`, or the ROS 2 node), the work is split
across processes. One process per camera reads it through ffmpeg into shared
memory, and records its stream as is when recording is on. A single pipeline
process runs the detector and NLF on the GPU and the inverse kinematics on the
CPU. It always takes the latest image of every camera: when processing falls
behind, frames are skipped rather than queued, so the delay does not grow. A 3D
viewer runs in the browser. See [Live capture](live.md).

**Offline** (`run_pipeline.py --dataset ...`), the same stages run one after
the other over recorded videos, every frame is processed, and results are
written per trial, ready to be compared with motion capture. See
[Recorded data and evaluation](offline.md).

## Frames and units

- **World**: the frame the calibration defines, on the floor or at a robot's
  base, z up. Without a world pose, positions stay in the reference camera's
  frame.
- **Cameras**: OpenCV's convention, x right, y down, z forward. Calibrations
  store each camera's pose in the world; see
  [Data format and camera conventions](data-format.md).
- **Model**: segments built y up, as the ISB recommends, under a fixed rotation
  of the root joint; see [Outputs](outputs.md#frames).
- **Units**: metres and radians throughout; time is counted in frames at `fs`.

## Where it lives in the code

| Stage | Module | Main entry points |
|---|---|---|
| Cameras and calibration | [`rtcosmik.camera`](https://gepetto.github.io/rt-cosmik/api/camera.html) | `select_live_cameras`, `load_camera_parameters`, `Camera` |
| Landmarks in each view | [`rtcosmik.nlf`](https://gepetto.github.io/rt-cosmik/api/nlf.html) | `NLFEstimator`, `extract_views` |
| Fusion | [`rtcosmik.triangulation`](https://gepetto.github.io/rt-cosmik/api/triangulation.html) | `reconstruct_3d`, `triangulate_points` |
| Filtering | [`rtcosmik.filtering`](https://gepetto.github.io/rt-cosmik/api/filtering.html) | `MarkerFilter` |
| Model and inverse kinematics | [`rtcosmik.pipeline.solver`](https://gepetto.github.io/rt-cosmik/api/pipeline.html), [`rtcosmik.ik`](https://gepetto.github.io/rt-cosmik/api/ik.html), [`rtcosmik.human_model`](https://gepetto.github.io/rt-cosmik/api/human_model.html) | `HumanSolver` |
| Live pipeline | [`rtcosmik.pipeline.pipeline`](https://gepetto.github.io/rt-cosmik/api/pipeline.html) | `PipelineProcess` |
| Recording | [`rtcosmik.saver`](https://gepetto.github.io/rt-cosmik/api/saver.html) | `Recorder` |
| Viewer | [`rtcosmik.viewer`](https://gepetto.github.io/rt-cosmik/api/viewer.html) | `Viewer`, `ComfiScene` |
