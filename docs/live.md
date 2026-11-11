# Live capture

Running RT-COSMIK on cameras: which cameras it opens, how it records, how to
test the live path without a rig, and how to read its timings. Calibrate the
cameras first (see the [README](own-cameras.md) and
[data-format.md](data-format.md)).

```bash
python3 scripts/python/core/run_pipeline.py --online --cameras 0 2 4 6
```

Cameras are opened through **ffmpeg**, not OpenCV: `cv2.VideoCapture` ignores
`CAP_PROP_BUFFERSIZE` on the V4L2 backend, so frames queue in the driver and
arrive late, and it cannot record without a decode/re-encode cycle. ffmpeg needs
to be on `PATH`.

Camera calibration comes from `settings.cam_calib_path`, or `--cam-params`.

## Which cameras are used

`--cameras` (default: `cameras` in `settings.py`) names **calibrated** cameras:
the `camera_<id>` of the calibration files. RT-COSMIK lists the attached
devices, works out which calibrated camera each one is, and opens only the ones
requested, in the requested order:

- With a `cameras.yaml` in the calibration (cams_calibration writes one), each
  device is recognised by its USB port, so a camera keeps its calibration when
  the rig is recabled or the computer rebooted.
- Without it, the id is the `/dev/video<id>` number.
- Anything else plugged in, such as a laptop's built-in webcam, stays closed.
- A requested camera that is not attached stops the run, with the list of what
  is attached.

The first camera is the reference frame. If it has no world pose and another
requested camera has one, that camera becomes the reference, so positions stay
in room coordinates (see [data-format.md](data-format.md#providing-extrinsics)).

Moving a camera to a new place in the room cannot be detected: recalibrate.

## Recording

Set in `settings.py`:

| | |
|---|---|
| `SAVE_CSV` | markers and joint angles, with a frame counter per camera |
| `SAVE_VID` | one `camera_<id>.mkv` per camera |
| `record_on_start` | begin recording immediately, for headless or scripted runs |
| `SAVE_DIR` | where they go (`output/<no_trial>`) |

Video is a **stream copy of the camera's own MJPEG**: no decode, no re-encode,
so recording is nearly free and the file is what the sensor produced.

Recording starts with the run by default. With `record_on_start = False`,
press **`s`** to start and **`q`** to stop. The
listener reads the terminal, so it works over SSH — unlike a keyboard hook,
which needs an X display and fails on a headless or remote session.

## Testing it without a rig

Recordings can be replayed through the *live* path — the same camera processes,
barrier, shared buffers and pipeline, with files standing in for devices:

```bash
python3 scripts/python/core/run_pipeline.py --online \
  --replay     <dataset>/videos/<participant>/<task> \
  --cam-params <dataset>/cam_params/<participant> \
  --subject    <dataset>/metadata/<participant>.yaml \
  --cameras 0 2 4 6
```

Playback is paced at the recording's own frame rate, so this shows whether the
pipeline *keeps up* rather than just how fast it can consume a file. The sources
also hold their first frame until the model is calibrated, because a real
subject stands still for that — without it the trial runs on during calibration,
and the model gets scaled from whatever pose it lands on.

It exercises the software path, not the capture hardware: every file source is
always ready, so the barrier never actually waits and real inter-camera skew
stays invisible.

## Reading the timings

The pipeline prints a line every couple of seconds while it runs:

```
[TIME]  37.2 turns/s | loop  26.9 ms (pose 18.9, ik  5.1) | kept  93% of camera frames
```

and a per-stage summary on exit. `wait` is time blocked waiting for every camera
to publish a new frame, so a late camera shows up there rather than in `pose`.

`kept %` is how many camera frames were processed. To see *where* frames were
lost — a warm-up cost, or a recurring stall:

```bash
python3 scripts/python/eval/frame_drops.py output/<run>/markers.csv
```

## Related entry points

`run_nlf_inference.py` (pose estimation only) and `run_triangulation.py`
(through triangulation) accept the same trial arguments, which is handy for
isolating a stage.
