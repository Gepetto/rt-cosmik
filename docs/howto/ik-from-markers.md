# Run the IK on your own markers

RT-COSMIK's model and inverse kinematics can be used on their own, on 3D
landmarks from any source: a run you want to reprocess with other settings,
another markerless system, or motion capture. This is how the paper obtained
its reference: motion capture markers processed through the same model and
inverse kinematics as the cameras.

## What the markers must be

- The 43 landmarks of `settings.marker_names`, listed in
  [Outputs](../outputs.md#markerscsv), with these names. Motion capture markers
  at other places need to be renamed, or computed from the ones you have.
- In metres, in a world frame with z up.
- At the frame rate the solver was generated for, `fs` (40 Hz by default).
  Resample them, or change `fs` and regenerate the solver (see
  [Inverse kinematics](../inverse-kinematics.md)).
- Starting with the person standing still: the first frame calibrates the
  model.

## The loop

`HumanSolver` builds the model, calibrates it on the first frame and solves
every following one. This reads a `markers.csv` written by RT-COSMIK and writes
the joint angles in the same layout as `joint_angles.csv`:

```python
import numpy as np
import pandas as pd

from rtcosmik.config_loader import settings
from rtcosmik.pipeline.solver import HumanSolver

markers = pd.read_csv("output/2112/RobotWelding/4cam_mhe_acados/markers.csv")
names = list(settings.marker_names)

solver = HumanSolver(settings, height=1.77, weight=62.0, gender="f")
poses = []
for _, row in markers.iterrows():
    frame = {name: row[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(float)
             for name in names}
    poses.append(solver.solve(frame))          # calibrates on the first frame

angles = pd.DataFrame(np.array(poses), columns=settings.joint_angles_names)
angles.to_csv("joint_angles_from_markers.csv", index=False)
```

On the [sample trial](../getting-started.md), this reproduces the pipeline's
own `joint_angles.csv` to numerical precision: the landmarks RT-COSMIK saves
are the ones its solver was given.

Settings such as `ik_type`, `mhe_profile` or `N` apply here as everywhere, so
the same markers can be solved with another configuration and compared, for
instance with `compare_to_mocap.py` (see
[Recorded data and evaluation](../offline.md)).
