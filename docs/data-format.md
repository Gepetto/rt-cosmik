# Data format and camera conventions

RT-COSMIK reads recordings and calibrations in one layout, the one of the
[COMFI dataset](https://doi.org/10.5281/zenodo.17223909). The
[sample trial](../README.md#quick-start) is a small example of it, and
[cams_calibration](https://github.com/Gepetto/cams_calibration) writes
calibrations in it directly.

## Layout

```
<dataset>/cam_params/<participant>/                      calibration
          ├── intrinsics/camera_<i>_intrinsics.yaml      K, D (OpenCV FileStorage)
          └── extrinsics/                                either source, see below
              ├── cam_to_world/camera_<i>/camera_<i>_extrinsics.yaml
              └── cam_to_cam/camera_<a>_to_camera_<b>.yaml
<dataset>/videos/<participant>/<task>/camera_<i>.mp4     synchronised video
<dataset>/metadata/<participant>.yaml                    id, height, weight, gender
```

`metadata/<participant>.yaml` describes the person, so that the model is scaled
to them:

```yaml
id: 2112
height: 1.77      # m
weight: 62.0      # kg
gender: female    # or male
```

Camera ids are free integers, the same in every file name: `camera_0`,
`camera_2`, ... The videos of one trial must be synchronized frame by frame.

COMFI also provides `robot/` (robot base pose and joint states), `mocap/` (the
marker-based reference) and `forces/`. RT-COSMIK does not need them to run; the
3D viewer draws the robot from them, and `compare_to_mocap.py` reads `mocap/` (see
[offline.md](offline.md)).

## Camera pose convention

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

## Providing extrinsics

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
it. Live runs make an anchored camera the reference when the first camera
requested has no world pose.

Without any anchor the reference camera's own frame becomes the world frame, with
a warning. Joint angles are unaffected, since they depend only on relative
geometry, but positions are then in camera coordinates rather than room
coordinates.

## Calibrating a rig

[cams_calibration](https://github.com/Gepetto/cams_calibration) produces this
layout directly, and installs it here:

```bash
python3 scripts/calibrate_cameras.py --cameras 0 2 4 6 --install
python3 scripts/set_world_frame.py   --cameras 0 2 4 6 --install
```

It also records a `cameras.yaml` naming the USB port behind each camera id.
Live runs match on it instead of trusting the v4l2 index, so a recabled rig is
remapped rather than silently paired with the wrong calibration. See
[live.md](live.md#which-cameras-are-used).

