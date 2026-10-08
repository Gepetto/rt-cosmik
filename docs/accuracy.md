# Accuracy

RT-COSMIK was evaluated on [COMFI](https://doi.org/10.5281/zenodo.17223909): 18
participants performing six demanding industrial tasks (screwing, polishing,
overhead work, lifting, and polishing and welding with a collaborative robot),
against marker-based motion capture processed through the same model and the
same inverse kinematics.

![The six tasks of the evaluation. Top: RT-COSMIK's estimate drawn over a camera image. Bottom: the same instant in 3D, with the motion capture reference in black, RT-COSMIK in green, and a 2D-keypoint baseline in yellow.](assets/tasks.jpg)

*Top: RT-COSMIK's estimate (NLF-3D) drawn over one camera image. Bottom: the
same instant in 3D, with the motion capture reference in black and a
2D-keypoint baseline (RTMPose+LSTM) in yellow.*

## Results

| Cameras | Joint angles (RMSE) | Position (marker error) | Hand–robot distance (RMSE) | REBA score (mean abs. error) | Processing rate |
|---|---|---|---|---|---|
| 4 | **9.7°** | **53 mm** | **24 mm** | **0.70** | 43 Hz |
| 2, facing each other | 10.0° | 61 mm | 32 mm | 0.71 | 58 Hz |
| 2, side by side | 10.0° | 111 mm | 43 mm | 0.72 | 58 Hz |
| 1 | 10.5° | 124 mm | 58 mm | 0.77 | 73 Hz |

Means across participants, whole body, on an RTX 4500 Ada GPU and an Intel
i9-14900K. On the same data, a 2D-keypoint baseline (RTMPose with OpenCap's
marker augmenter) reached 13.1° with four cameras.

Errors are lowest on the trunk and legs (3 to 4° for lumbar and knee flexion)
and highest on elbow pronation–supination and ankle inversion–eversion, axial
rotations that barely move the surface of the body. Most of the error is an
offset rather than noise, so scores computed relative to a neutral posture,
such as REBA, are less affected than absolute angles.

The evaluation used a 7-frame horizon and a 5 Hz filter (the defaults are 10
frames and 10 Hz), with the thoracic and wrist joints frozen for the comparison
with the baseline. Details are in the paper.

## Deployment guidelines

What the evaluation suggests when setting up a workcell:

- **For ergonomics, one camera is almost enough.** It lost 0.7° of joint-angle
  accuracy and 0.07 points of REBA error against four cameras.
- **For positions relative to a robot, place the cameras first.** Joint angles
  depend on relative positions and barely change with the camera layout, but
  the position of the body does: two facing cameras come close to four, whereas
  two cameras side by side double the position error, which is mostly along
  their common viewing direction.
- **The pose estimator sets the rate.** The inverse kinematics takes a few
  milliseconds per frame; the estimator takes most of the rest.
- **Keep the moving-horizon inverse kinematics.** It was 1.4° more accurate
  than solving each frame on its own, for 1.7 ms more per frame.
