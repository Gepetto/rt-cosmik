# Publish to ROS 2

[rtcosmik_ros](https://github.com/Gepetto/rtcosmik_ros) is a ROS 2 node that
runs RT-COSMIK's live pipeline and publishes its results for RViz and for the
rest of a robot's software. It is a thin wrapper: cameras, calibration,
inverse kinematics and recording are RT-COSMIK's own, configured by the same
`settings.py`. Its [README](https://github.com/Gepetto/rtcosmik_ros#readme) has
the details; this page is the short version.

## Build and run

RT-COSMIK's Docker image has ROS 2 Humble. Clone `rtcosmik_ros` next to
`rt-cosmik` before starting the container: `docker/run.sh` mounts it at
`/root/workspace/ros_ws/src/rtcosmik_ros`. Then, in the container:

```bash
cd /root/workspace/ros_ws
colcon build --packages-select rtcosmik_ros
source install/setup.bash
ros2 launch rtcosmik_ros start.launch.py
```

The node opens the cameras of `settings.cameras`, as `run_pipeline.py --online`
does (see [Live capture](../live.md#which-cameras-are-used)), and starts RViz
once the model is calibrated on the person in front of the cameras.

## Published topics

| Topic | Type | Contents |
|---|---|---|
| `/rtcosmik/q` | `Float64MultiArray` | The model's configuration, as the columns of `joint_angles.csv` |
| `/rtcosmik/joint_states` | `JointState` | The joint angles, for `robot_state_publisher` |
| `/rtcosmik/markers` | `MarkerArray` | The fused landmarks |
| `/rtcosmik/collision_poses` | `PoseArray` | Poses of the body's collision capsules |
| `/rtcosmik/collision_markers` | `MarkerArray` | The same capsules, drawable |

The node also writes a URDF of the model scaled to the person and hands it to
`robot_state_publisher`, so RViz shows a body of the right size.

## Without cameras

Recordings replay through the node as if they were cameras. The
[sample trial](../getting-started.md) works as is, with the default `cameras`:

```bash
ros2 launch rtcosmik_ros start.launch.py \
    replay_dir:=/root/workspace/rt-cosmik/data/comfi_sample/videos/2112/RobotWelding \
    cam_calib_path:=/root/workspace/rt-cosmik/data/comfi_sample/cam_params/2112
```

The person's height, mass and sex come from `settings.py`; set them to 1.77 m,
62 kg and `'f'` to match the sample's participant.
