## RT-COSMIK
***Real-Time - Constrained and Open Source Multibodied Inverse Kinematics***

RT-COSMIK is a cutting-edge open-source library for solving real-time constrained inverse kinematics problems for multibody systems. It is designed for robotics applications, offering robust integration with ROS and advanced features like real-time pipelines, MMpose, and LSTM-based motion prediction.

---

To generate the appropriate models, use:

```bash 
./scripts/bash/fetch_models.sh 
```

To install the toolbox and use the scripts files: 
```bash 
pip install -e .
```

## Quick start: offline evaluation on a recorded dataset

Run the pipeline over recorded video and compare the result against reference
mocap. The example below uses participant `1012`, task `Lifting`.

### 1. Install and fetch the models

```bash
pip install -e .
./scripts/bash/fetch_models.sh
```

The human model comes from `example-robot-data`. Its thorax visual is misplaced
upstream (the chest renders below the thoracic joint and overlaps the abdomen,
increasingly so for taller subjects); the fix lives on the
`fix/thorax-visual-scale-and-origin` branch of the fork. Kinematics are
unaffected either way, so this only matters for how the model looks in the
viewer.

`fetch_models.sh` downloads the NLF and YOLO weights and exports one TensorRT
detector engine per supported camera count (2 to 6). Engines are built
non-dynamic, so the batch size is fixed at export time and the pipeline picks the
engine matching the cameras in use. Build a subset with `BATCHES="2 4"`.

### 2. Data layout

Any dataset in this layout works, not one in particular:

```
<dataset>/cam_params/<participant>/                      calibration
          ├── intrinsics/camera_<i>_intrinsics.yaml      K, D (OpenCV FileStorage)
          └── extrinsics/                                either source, see below
              ├── cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
              └── cam_to_cam/camera_<a>_to_camera_<b>.yaml
<dataset>/videos/<participant>/<task>/camera_<i>.mp4     synchronised video
<dataset>/metadata/<participant>.yaml                    id, height, weight, gender
```

#### Camera pose convention

**RT-COSMIK expects the pose of the camera in the world frame.** One convention,
everywhere, for every entry point:

```
R  3x3  the camera's orientation in world coordinates
        (its columns are the camera's x, y, z axes expressed in the world frame)
T  3x1  the camera's position in world coordinates, in metres
```

Equivalently, the pair maps a point from camera coordinates into world
coordinates:

```
p_world = R @ p_cam + T
```

**How to check you have it the right way round.** `T` is the camera's physical
position in the room, so read it back and see whether it describes where the
camera actually is. A correct 4-camera rig looks like this:

```
camera_0: T = [-0.82 -3.02  1.11]     all four at 1.11 m height,
camera_2: T = [-0.00 -2.96  1.11]     two at y ~ -3, two at y ~ +2.3,
camera_4: T = [-0.61  2.31  1.11]     i.e. facing each other across
camera_6: T = [ 0.19  2.32  1.11]     a capture volume ~5 m deep
```

Those are metres from the world origin, and they match the room. If instead `T`
comes out near zero, or at an implausible height, the transform is inverted.
Getting this backwards raises no error: the skeleton is simply reconstructed in
the wrong place and orientation.

Read your own back with:

```python
from rtcosmik.camera.cam_utils import describe_camera_placement
describe_camera_placement("<dataset>/cam_params/<participant>")
```

**Converting from OpenCV.** `cv2.solvePnP` and most aruco helpers give you the
*opposite* transform — the world/marker expressed in camera coordinates
(`p_cam = R_cv @ p_world + t_cv`). Invert it before saving:

```python
R = R_cv.T
T = -R_cv.T @ t_cv
```

#### Providing extrinsics

Poses come from either of two sources, whichever the calibration produced. The
loader picks automatically, and `load_camera_parameters(..., extrinsics_source=)`
forces one.

**A world pose per camera** — what a fit against shared motion-capture markers
produces. Each camera is placed independently, so error does not accumulate.
This is preferred when available.

```
extrinsics/cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
```
```yaml
camera_extrinsics:
  frame_from: camera_0
  frame_to: world
  rotation_matrix: [[...], [...], [...]]   # R, camera orientation in world
  translation_vector: [tx, ty, tz]         # T, camera position in world, metres
```

**Stereo pairs plus one anchor** — what a checkerboard calibration produces, and
the usual online case: a checkerboard gives intrinsics and pairwise poses, and a
single aruco marker fixes one camera in the world.

```
extrinsics/cam_to_cam/camera_<a>_to_camera_<b>.yaml   OpenCV stereoCalibrate output
extrinsics/cam_to_world/camera_<ref>/...              the anchor, reference camera only
```

Pairs hold `R`, `T` in `cv2.stereoCalibrate`'s own convention (`p_b = R @ p_a + T`),
so they are saved exactly as OpenCV writes them — no inversion. They are chained
from the reference camera, in either direction, by the shortest path. Only the
**reference** camera needs a world pose; every other camera is placed relative to
it.

Without any anchor the reference camera's own frame becomes the world frame, with
a warning. Joint angles are unaffected, since they depend only on relative
geometry, but positions are then in camera coordinates rather than room
coordinates.

### 3. Run one trial

```bash
python3 scripts/python/core/run_pipeline.py \
    --dataset /path/to/COMFI --participant 1012 --task Lifting
```

Results land in `output/1012/Lifting/<variant>/`, mirroring the dataset layout.
The variant names the settings that distinguish one run from another - the
camera count and the IK method - so switching solver or cameras writes a new
directory instead of overwriting the previous run:

```
output/1012/Lifting/4cam_mhe_fatrop/     # settings.ik_type = "mhe", mhe_backend = "fatrop"
output/1012/Lifting/4cam_sbs/            # settings.ik_type = "sbs"
output/1012/Lifting/1cam_mhe_fatrop/     # --cameras 0
```

Each directory holds:

| file | contents |
|---|---|
| `joint_angles.csv` | 43 DoF per frame, using the standard RT-COSMIK column names |
| `markers.csv`      | triangulated 3D markers per frame, in metres, world frame |

Alongside them, `run_info.json` records the full configuration (cameras,
subject, IK type and solver settings, filter, frame counts, model root frame),
which the evaluation tools read. Only the discriminating knobs go in the
directory name; everything else is recorded there.

Useful flags:

```bash
--cameras 0 2          # use a subset; the first is the triangulation reference frame
                       # a single camera works too (NLF's monocular 3D is used)
--out DIR              # write somewhere other than output/<participant>/<task>
--no-save              # visualise only
```

Meshcat prints a viewer URL at startup for live 3D inspection.

Fully explicit paths work for data outside the shorthand layout:

```bash
python3 scripts/python/core/run_pipeline.py \
    --cam-params CAL/S03 --trial-dir VIDEO/S03/Lifting --subject META/S03.yaml
```

### 4. Compare against mocap

One script does the whole evaluation: error tables, figures, and a 3D replay.
A single camera is supported - there is nothing to triangulate, so the metric
3D pose NLF regresses from that view is used directly.

```bash
# produce one run per setup (variant directories keep them apart)
for cams in "0" "0 2" "0 2 4 6"; do
  python3 scripts/python/core/run_pipeline.py --dataset /path/to/COMFI \
      --participant 1012 --task Lifting --cameras $cams
done

# compare them all against mocap: tables, figures and the 3D view
python3 scripts/python/eval/compare_to_mocap.py \
    --reference /path/to/COMFI/mocap/aligned/1012/Lifting \
    1cam=output/1012/Lifting/1cam_mhe_fatrop \
    2cam=output/1012/Lifting/2cam_mhe_fatrop \
    4cam=output/1012/Lifting/4cam_mhe_fatrop \
    --plots output/1012/eval_Lifting --meshcat
```

Any labels work, so the same command compares IK methods instead of camera
counts:

```bash
python3 scripts/python/eval/compare_to_mocap.py \
    --reference /path/to/COMFI/mocap/aligned/1012/Lifting \
    sbs=output/1012/Lifting/4cam_sbs \
    mhe=output/1012/Lifting/4cam_mhe_fatrop \
    --plots output/1012/eval_ik --meshcat
```

Before anything is compared, the runs are **time-aligned** to the mocap. The
cameras are synchronised with each other but not with the mocap, so a single
offset covers them all: it is estimated per run by correlating knee flexion
against the reference, and the median is applied to every modality. The
estimates and the applied lag are printed.

**Tables** print per-joint and per-marker error with one column per run, each
ending with the mean and median across all joints or all markers.

**Figures** (`--plots DIR`, which also receives `errors.csv` with the same
numbers for a spreadsheet):

| figure | shows |
|---|---|
| `joint_angle_rmse.png`            | error per degree of freedom, plus the mean over all joints |
| `marker_error.png`                | 3D error per marker, plus the mean over all markers |
| `joint_angle_trajectories.png`    | every joint angle over time, each panel captioned with its own RMSE |
| `marker_error_distribution.png`   | spread of marker error per modality |

The bar charts carry a bold `MEAN (all …)` row at the top, so a modality can be
judged as a whole before reading the per-item breakdown.

**3D replay** (`--meshcat`) shows every modality at once, each drawn as the
human model it solved on, tinted with the colour it has in the tables and
figures. Each run records its calibrated model in `run_info.json`, so the body
shown is the one the IK used - no external model file is needed. The reference
contributes its markers, the ground truth being compared against. Models are
semi-transparent so overlapping bodies stay readable. Open the printed URL in a
browser. Playback is stepped by hand from the terminal so you can stop on any
instant:

| key | action |
|---|---|
| `space` | play / pause |
| `n` / `p` | one frame forward / back |
| `f` / `b` | jump 25 frames forward / back |
| `[` / `]` | slower / faster |
| `r` | back to the first frame |
| `q` | quit |

Every modality keeps the same colour and label across the tables, the figures
and the 3D view, so a colour means the same thing everywhere.

`joint_angles.csv` uses the same column names and ordering as the reference, so
the two line up without renaming. The free-flyer needs one extra step:
RT-COSMIK's human model carries a fixed rotation on its root joint while the
reference URDF does not, so the two base frames differ. Each run records its
root placement in `run_info.json` and the comparison removes it before
reporting, so the free-flyer is compared like for like.

### 5. Sweep several trials

There is no batch script; a shell loop does the job.

```bash
for p in $(ls /path/to/COMFI/videos); do
  for t in $(ls /path/to/COMFI/videos/$p); do
    python3 scripts/python/core/run_pipeline.py \
        --dataset /path/to/COMFI --participant $p --task $t || echo "FAILED $p/$t"
  done
done
```

### Related entry points

`run_nlf_inference.py` (pose estimation only) and `run_triangulation.py`
(through triangulation) accept the same trial arguments, which is handy for
isolating a stage.

## Citing RT-COSMIK


## License
BSD 2-Clause License

Copyright (c) 2024, LAAS-CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Project Status
RT-COSMIK is currently under active development. Contributions and feedback are welcome. 

