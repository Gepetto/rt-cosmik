# Configuration

All configuration lives in [`settings.py`](https://github.com/Gepetto/rt-cosmik/blob/main/settings.py),
at the root of the repository. Command-line arguments only say which data to
process: which trial, which calibration, which cameras. Another settings file
can be used for a run by pointing `RTCOSMIK_SETTINGS_PATH` at it. If the file
cannot be found or fails to load, RT-COSMIK logs a warning and falls back to the
defaults below, on the CPU: after editing it, check the log for that warning.

Settings marked *computed* are derived from the others when the settings are
loaded; change the ones they are derived from instead.

## Cameras

| Setting | Default | Meaning |
|---|---|---|
| `cameras` | `(0, 2, 4, 6)` | Calibrated cameras to use, in order; the first is the reference. `--cameras` overrides it. |
| `fs` | `40` | Camera frame rate (Hz). The acados solver bakes its time step from it: regenerate it after a change. |
| `width`, `height` | `1280`, `720` | Image size the cameras stream. |
| `fourcc` | `"MJPG"` | Camera stream format: `"MJPG"`, or raw `"YUYV"`. |
| `cam_calib_path` | *computed*: `config/cam_params` | Calibration of live runs. `--cam-params` overrides it. |

## Person

| Setting | Default | Meaning |
|---|---|---|
| `human_height` | `1.80` | Height (m) of the person in front of the cameras, for live runs. |
| `human_weight` | `70.0` | Mass (kg). |
| `human_gender` | `'m'` | `'m'` or `'f'`: selects the anthropometric tables used to scale the model. |

Offline runs of a dataset read the person from the dataset's metadata instead
(see [Data format](data-format.md)).

## Recording

| Setting | Default | Meaning |
|---|---|---|
| `no_trial` | `"test"` | Name of the live recording: results go to `output/<no_trial>/`. |
| `SAVE_CSV` | `True` | Live runs save the landmarks and the joint angles. |
| `SAVE_VID` | `True` | Live runs save each camera's video, as the camera streamed it. |
| `record_on_start` | `True` | Start recording with the run; otherwise `s` starts and `q` stops it. |
| `record_hotkeys` | `False` | Also listen to `s` and `q` through the desktop keyboard (needs a display); the terminal keys work regardless. |

## Pose estimation

| Setting | Default | Meaning |
|---|---|---|
| `yolo_model` | `"yolov10n"` | Person detector. Run `fetch_models.sh` again after changing it. |
| `yolo_conf` | `0.2` | Minimum detection confidence. |
| `yolo_imgsz` | `640` | Detector input size (pixels). |
| `nlf_indices` | 43 vertex indices | Points of the SMPL-X template NLF is queried at, one per landmark of `marker_names`. |
| `nlf_path`, `cano_path`, `yolo_path` | *computed* | Model files under `weights/`. |
| `device` | *computed* | `"cuda:0"` when a GPU is available, else `"cpu"`. |

## Filtering

| Setting | Default | Meaning |
|---|---|---|
| `order` | `4` | Order of the Butterworth filter applied to the fused landmarks. |
| `cutoff_freq` | `10` | Cutoff frequency (Hz): lower is smoother, higher reacts faster. |
| `filter_type` | `"lowpass"` | Filter type. |

## Inverse kinematics

| Setting | Default | Meaning |
|---|---|---|
| `ik_type` | `"mhe"` | `"mhe"`: moving-horizon estimation; `"sbs"`: one frame at a time. See [Inverse kinematics](inverse-kinematics.md). |
| `mhe_backend` | `"acados"` | Solver of the moving-horizon problem: `"acados"` or `"fatrop"`. |
| `mhe_profile` | `"realtime"` | Speed/accuracy trade-off: `"realtime"` or `"accurate"`. |
| `N` | `10` | Horizon: number of frames the moving-horizon problem fits at once. |
| `cost_weights` | `[1, 1e-3, 1e-5]` | Weights of the landmark, state and acceleration costs. |
| `ik_code` | `"c"` | fatrop only: `"c"` loads the generated, compiled solver; `"python"` runs the CasADi function directly. |
| `acados_export_dir` | `None` | Where generated solvers are kept; `None` means `ocp/` in the repository. `RTCOSMIK_OCP_DIR` overrides both. |
| `acados_source_dir` | `None` | acados installation; `None` reads `ACADOS_SOURCE_DIR`. |

The solver must be regenerated after a change to what it is built from: `fs`,
`N`, the profile, or the tracked landmarks. `run_ocp_codegen.py --check` says
whether it is up to date, and the pipeline refuses a stale one.

## Landmarks and outputs

| Setting | Meaning |
|---|---|
| `marker_names` | The 43 landmarks, in the order the pose estimator returns them. |
| `keys_to_track_list` | The landmarks the inverse kinematics fits; all of them by default. |
| `joint_angles_names` | Column names of `joint_angles.csv`, in the order of the model's configuration. |

## Viewer

| Setting | Default | Meaning |
|---|---|---|
| `viewer_scene` | `True` | Offline replays of a dataset trial draw the room around the body: floor, cameras, and for COMFI's robot tasks the table and the robot in its recorded state. |
