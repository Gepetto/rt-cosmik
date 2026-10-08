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

Calibration uses [cams_calibration](https://github.com/Gepetto/cams_calibration).
Clone it next to `rt-cosmik` before starting the container, so that
`docker/run.sh` mounts it. Then, in the container:

```bash
cd /root/workspace/cams_calibration
# checkerboard: each camera, then the camera pairs
python3 scripts/calibrate_cameras.py --cameras 0 2 4 6 --install
# wand: the world frame on the floor (add --robot to put it at the robot base)
python3 scripts/set_world_frame.py --cameras 0 2 4 6 --install
```

`--install` writes the calibration into `config/cam_params/`, where RT-COSMIK
reads it. Camera ids are the `/dev/video<id>` numbers when you calibrate; the
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
