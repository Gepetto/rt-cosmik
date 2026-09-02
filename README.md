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
          └── extrinsics/cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
<dataset>/videos/<participant>/<task>/camera_<i>.mp4     synchronised video
<dataset>/metadata/<participant>.yaml                    id, height, weight, gender
```

Camera poses are read from `cam_to_world`, which maps a point in the camera
frame into the world frame (`p_world = R @ p_cam + T`).

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
    --markers --plots output/1012/eval_Lifting \
    --meshcat --urdf /path/to/COMFI/metadata/urdf/1012_scaled.urdf
```

Any labels work, so the same command compares IK methods instead of camera
counts:

```bash
python3 scripts/python/eval/compare_to_mocap.py \
    --reference /path/to/COMFI/mocap/aligned/1012/Lifting \
    sbs=output/1012/Lifting/4cam_sbs \
    mhe=output/1012/Lifting/4cam_mhe_fatrop \
    --markers --plots output/1012/eval_ik --meshcat
```

**Tables** print per-joint and per-marker error with one column per run, plus a
summary. `--csv table.csv` writes the same numbers for a spreadsheet.

**Figures** (`--plots DIR`):

| figure | shows |
|---|---|
| `joint_angle_rmse.png`            | error per degree of freedom, one bar per setup |
| `marker_error.png`                | 3D error per marker, one bar per setup |
| `joint_angle_trajectories.png`    | representative joint angles over time against mocap |
| `marker_error_distribution.png`   | spread of marker error per setup |

**3D replay** (`--meshcat`) shows every modality at once: mocap markers in
white, each run in its own colour, and with `--urdf` a skeleton per source posed
from its joint angles, so marker error and pose error can be judged together.
Open the printed URL in a browser. Playback is stepped by hand from the terminal
so you can stop on any instant:

| key | action |
|---|---|
| `space` | play / pause |
| `n` / `p` | one frame forward / back |
| `f` / `b` | jump 25 frames forward / back |
| `[` / `]` | slower / faster |
| `r` | back to the first frame |
| `q` | quit |

`--play` runs straight through instead, `--loop` repeats, and
`--start/--end/--step/--fps` restrict or retime the replay.

`joint_angles.csv` uses the same column names and ordering as the reference, so
the two line up without renaming. The free-flyer needs one extra step:
RT-COSMIK's human model carries a fixed rotation on its root joint while the
reference URDF does not, so the two base frames differ. Each run records its
root placement in `run_info.json` and the comparison removes it before
reporting, so the free-flyer is compared like for like.
`--no-align-freeflyer` disables that.

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

