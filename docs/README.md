<p align="center">
  <img src="assets/logo.png" alt="COSMIK logo" width="120">
</p>

# RT-COSMIK

**Real-time, markerless, whole-body biomechanics from ordinary cameras.**

RT-COSMIK estimates the joint angles and the global pose of a whole-body
biomechanical model from synchronized RGB cameras, live, at the camera rate: 40
Hz with four 720p webcams on a single workstation. It was built for the
workplace: continuous ergonomic monitoring of operators, and human–robot
collaboration, where the robot needs to know where the operator's body is.

![Left: a camera view of a person welding with a collaborative robot, with RT-COSMIK's estimated body model drawn over them. Right: the same motion in RT-COSMIK's 3D viewer.](assets/hero.gif)

## Where to start

- **Try it without cameras.** The [quick start](getting-started.md) installs
  RT-COSMIK with Docker and runs it on a 20-second recording of the COMFI
  dataset, then compares the result with motion capture.
- **Set up your own cameras.** [Your own cameras](own-cameras.md) covers what to
  buy, where to place the cameras, how to calibrate them, and how to run and
  record live.
- **Understand what it computes.** The [overview](overview.md) walks through the
  pipeline, the human model, the landmarks and the frames the results are
  expressed in.
- **Build on it from Python.** The how-to guides show how to
  [plug in another pose estimator](howto/pose-estimator.md) and
  [run the inverse kinematics on your own markers](howto/ik-from-markers.md);
  the [API reference](https://gepetto.github.io/rt-cosmik/api/) documents every module.

## Related projects

- [rt-cosmik](https://github.com/Gepetto/rt-cosmik): the code, and the place for
  questions and bug reports, through GitHub issues.
- [cams_calibration](https://github.com/Gepetto/cams_calibration): calibrates a
  camera rig and installs the result into RT-COSMIK.
- [rtcosmik_ros](https://github.com/Gepetto/rtcosmik_ros): the ROS 2 node that
  runs the live pipeline and publishes its results.
- [COMFI](https://doi.org/10.5281/zenodo.17223909): the multimodal industrial
  dataset RT-COSMIK was validated on.

To cite RT-COSMIK, see the
[README](https://github.com/Gepetto/rt-cosmik#citing-rt-cosmik).
