# Recorded data and evaluation

RT-COSMIK runs the same pipeline on recorded videos as on live cameras, one
trial at a time, and compares the result with a motion capture reference. The
examples below use COMFI participant `1012`, task `Lifting`; the
[sample trial](../README.md#quick-start) works the same way with participant
`2112`, task `RobotWelding`. Recordings must follow the
[data format](data-format.md).

## Run one trial

```bash
python3 scripts/python/core/run_pipeline.py \
    --dataset /path/to/COMFI --participant 1012 --task Lifting
```

Results land in `output/1012/Lifting/<variant>/`, mirroring the dataset layout.
The variant names the settings that distinguish one run from another - the
camera count and the IK method - so switching solver or cameras writes a new
directory instead of overwriting the previous run:

```
output/1012/Lifting/4cam_mhe_acados/     # settings.ik_type = "mhe", mhe_backend = "acados" (default)
output/1012/Lifting/4cam_mhe_fatrop/     # mhe_backend = "fatrop"
output/1012/Lifting/4cam_sbs/            # settings.ik_type = "sbs"
output/1012/Lifting/1cam_mhe_acados/     # --cameras 0
```

Each directory holds:

| file | contents |
|---|---|
| `joint_angles.csv` | pelvis position and orientation, then the 36 joint angles, per frame |
| `markers.csv`      | the fused 3D landmarks per frame, in metres, world frame |

Alongside them, `run_info.json` records the full configuration (cameras,
subject, IK type and solver settings, filter, frame counts, model root frame),
which the evaluation tools read. Only the discriminating knobs go in the
directory name; everything else is recorded there. Columns, units and frames are
detailed in [outputs.md](outputs.md).

Useful flags:

```bash
--cameras 0 2          # use a subset; the first is the triangulation reference frame
                       # a single camera works too (NLF's monocular 3D is used)
--out DIR              # write somewhere other than output/<participant>/<task>
--no-save              # visualise only
```

Meshcat prints a viewer URL at startup for live 3D inspection. When the trial
comes from a dataset (`--dataset/--participant/--task`), the estimated body is
drawn in the room it was recorded in, as COMFI's own example viewer draws it:
the floor, the cameras used, and for `RobotPolishing`/`RobotWelding` the table
and the Franka Panda, which follows its recorded joint states frame by frame.
It is on by default (`viewer_scene` in `settings.py`); whatever the dataset
lacks is left out, and runs that are not a dataset trial (live cameras,
`--trial-dir`, `--videos`) show the body alone. For example:

```bash
python3 scripts/python/core/run_pipeline.py --dataset /path/to/COMFI \
    --participant 1012 --task RobotWelding --no-save
# open the printed http://127.0.0.1:7000/static/ URL
COMFI_ROOT=/path/to/COMFI python3 -m pytest tests/unit/viewer/test_comfi_scene.py
```

Fully explicit paths work for data outside the shorthand layout:

```bash
python3 scripts/python/core/run_pipeline.py \
    --cam-params CAL/S03 --trial-dir VIDEO/S03/Lifting --subject META/S03.yaml
```

## Compare against motion capture

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
    1cam=output/1012/Lifting/1cam_mhe_acados \
    2cam=output/1012/Lifting/2cam_mhe_acados \
    4cam=output/1012/Lifting/4cam_mhe_acados \
    --plots output/1012/eval_Lifting --meshcat
```

Any labels work, so the same command compares IK methods instead of camera
counts:

```bash
python3 scripts/python/eval/compare_to_mocap.py \
    --reference /path/to/COMFI/mocap/aligned/1012/Lifting \
    sbs=output/1012/Lifting/4cam_sbs \
    mhe=output/1012/Lifting/4cam_mhe_acados \
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

## Sweep several trials

There is no batch script; a shell loop does the job.

```bash
for p in $(ls /path/to/COMFI/videos); do
  for t in $(ls /path/to/COMFI/videos/$p); do
    python3 scripts/python/core/run_pipeline.py \
        --dataset /path/to/COMFI --participant $p --task $t || echo "FAILED $p/$t"
  done
done
```
