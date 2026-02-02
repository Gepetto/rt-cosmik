# cosmik_ros2

This package provides ROS 2 tooling for RT-COSMIK outputs (joint states, keypoints, markers, and segment poses via TF).

## What this package does

- Runs the RT-COSMIK pipeline while publishing ROS 2 topics.
- Publishes joint angles as `/human_RT_joint_angles` (`sensor_msgs/JointState`).
- Publishes keypoints as `/keypoints` (`visualization_msgs/MarkerArray`).
- Publishes augmented markers as `/lstm_markers` (`visualization_msgs/MarkerArray`).
- Broadcasts TF for the pelvis and optionally all model frames (segment poses).

## Prerequisites

- ROS 2 (tested with Humble/Iron).
- RT-COSMIK installed as a Python package (see root `setup.py`).
- Camera setup / configuration for RT-COSMIK.

## Build & source

From your ROS 2 workspace root:

```bash
colcon build --packages-select cosmik_ros2
source install/setup.bash
```

If you are actively developing RT-COSMIK as a Python package, ensure it is installed in the same environment:

```bash
pip install -e /path/to/RT-COSMIK
```

## Run the pipeline

```bash
ros2 launch cosmik_ros2 run_pipeline.launch.py
```

This launches the RT-COSMIK pipeline and publishes ROS 2 outputs.

## Configure ROS 2 outputs

Settings live in `settings.py` at the repo root. Important fields:

- `viewer = "ros2"`
- `ros2_world_frame` (default: `world`)
- `ros2_pelvis_frame` (default: `pelvis`)
- `ros2_publish_segments` (default: `True`)

## Verify outputs

- Joint states:
  ```bash
  ros2 topic echo /human_RT_joint_angles
  ```
- Keypoints:
  ```bash
  ros2 topic echo /keypoints
  ```
- TF tree:
  ```bash
  ros2 run tf2_tools view_frames
  ```

## Development checklist

1. Ensure `settings.viewer` is `"ros2"`.
2. Run the pipeline and verify joint state messages.
3. Enable `ros2_publish_segments` and verify TF frames in RViz.
4. Validate that joint names match the URDF names used in your robot.
5. Confirm units (meters/radians) and frame conventions.
