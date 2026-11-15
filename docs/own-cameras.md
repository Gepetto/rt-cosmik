# Your own cameras

From a rig of webcams to joint angles, live. This one-minute video shows the
whole setup: placing the cameras, calibrating them, setting the world frame,
and two operators tracked live next to a robot.

<video controls preload="metadata" width="100%" src="assets/how_to_setup.mp4"></video>

([Download the video](assets/how_to_setup.mp4) if it does not play here.)

## What you need

- **A computer** running Linux, with an NVIDIA GPU. 40 Hz with four cameras was
  measured with an RTX 4500 Ada and an Intel i9-14900K.
- **One to four USB webcams** streaming 1280×720 MJPEG at 40 fps. Other
  resolutions and rates work after setting `width`, `height` and `fs` in
  `settings.py`, and regenerating the solver (its time step depends on `fs`).
- **For calibration**, a printed checkerboard, and a wand (an ArUco marker on a
  stick) to set the world frame.

## 1. Place the cameras

Frame the whole working area in every view. Then place the cameras for what you
measure (see [Accuracy](accuracy.md)):

- **Joint angles only**, for ergonomics: one camera is almost as accurate as
  four.
- **Positions in the room**, for distances to a robot: put cameras on opposite
  sides of the workspace. Two facing cameras come close to four, whereas two
  cameras side by side double the position error.

## 2. Calibrate them

Calibration is part of the toolbox. Set the checkerboard (`checkerboard_rows`,
`checkerboard_columns`, `checkerboard_square_size`) and the wand
(`wand_marker_size`, `wand_tip_offset`) in `settings.py`
(see [Configuration](configuration.md#calibration)). Then, in the container:

```bash
# checkerboard: each camera, then the camera pairs
python3 scripts/python/calibration/calibrate_cameras.py --cameras 0 2 4 6
# wand: the world frame on the floor (add --robot to put it at the robot base)
python3 scripts/python/calibration/set_world_frame.py --cameras 0 2 4 6
```

In both, SPACE saves one image per camera and `q` finishes. Move the
checkerboard through the working area, keeping it visible to both cameras of
each adjacent pair: cameras are paired in the order given, and a pair needs at
least 6 shots seen by both. Then point the wand at each position of the world
frame, in order:

- on the floor: the origin, a point along x, and a point of the floor on the
  y side;
- with `--robot`: two points on opposite sides of the base, whose middle is the
  origin (y points towards the first), then two points whose middle is behind
  the base on its x axis.

The calibration is written to `config/cam_params/`, where RT-COSMIK reads it.
Only the first camera gets a world pose; the others are placed from it through
the camera pairs. `set_world_frame.py` prints where that camera is in the world
and saves `world_frame_check.png`, the world axes drawn in each view: check
both against the room. The images are kept in `output/calibration/`; add
`--from-images` to calibrate from them again, and `--no-show` without a
display.

### Without a checkerboard: from a person walking

When a checkerboard cannot be shown to two cameras at once, for instance cameras
that face each other across the workspace, the cameras can be placed from a
recording of one person walking through the working area for about a minute:

```bash
python3 scripts/python/calibration/calibrate_from_human.py --videos output/walk
```

It needs the intrinsics of each camera (from `calibrate_cameras.py`) and
replaces the camera pairs and the world frame: z up from the floor, x along the
first camera's viewing direction, origin on the floor under that camera. Run
`set_world_frame.py` afterwards for another world frame. A recording in which
the person does not walk across the area is refused.

This calibration is less accurate than the checkerboard: on COMFI it places
cameras 5 m apart within about 0.2 m and 1.6° of the reference calibration. Joint
angles are almost unaffected, positions in the room shift by a few centimetres
(see [Accuracy](accuracy.md)).

Camera ids are the `/dev/video<id>` numbers when you calibrate; the
calibration also records which USB port each camera is on, so a recabled rig is
recognised later (see [Live capture](live.md#which-cameras-are-used)). The
format of the calibration is described in
[Data format and camera conventions](data-format.md).

## 3. Say who is in front of the cameras

Set `human_height` (m), `human_weight` (kg) and `human_gender` (`'m'` or `'f'`)
in `settings.py`, and `cameras` to the ids you calibrated, for instance
`(0, 2, 4, 6)`.

## 4. Run live

```bash
python3 scripts/python/core/run_pipeline.py --online --cameras 0 2 4 6
```

Stand still and fully in view for a second when it starts: the model is fitted
to the person on the first frame. The 3D viewer is at
<http://127.0.0.1:7000/static/>.

## 5. Record

Live runs record the videos and the results to `output/<no_trial>/`. Name each
recording with `no_trial` in `settings.py`. Recording starts immediately; with
`record_on_start = False`, press `s` in the terminal to start and `q` to stop.
Recording, replays of recordings through the live path, and the timings the
pipeline prints are covered in [Live capture](live.md).

To publish the results to other robot software, see
[Publish to ROS 2](howto/ros2.md).
