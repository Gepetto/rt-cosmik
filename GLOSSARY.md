# RT-COSMIK

Real-time markerless estimation of whole-body biomechanical joint angles and global pose from several RGB cameras, validated against marker-based motion capture.

## Sensing

**View**:
The image stream of one calibrated camera.
_Avoid_: camera point of view, feed

**Calibration**:
The intrinsics of each view and the extrinsics relating all views to one world frame.

**Human-pose calibration**:
Calibration whose extrinsics are estimated from the observed person, without a checkerboard or wand.
_Avoid_: walking-person calibration, auto-calibration

**Reference calibration**:
The COMFI calibration obtained against the reference system (Söderkvist method); used only as ground truth, never in an evaluated pipeline.
_Avoid_: dataset calibration, mocap calibration

## Reconstruction

**Front end**:
The per-view estimator that turns an image into landmarks (NLF, FastSAM-3D, RTMPose+LSTM).
_Avoid_: estimator, HPE, backbone, modality

**Body model**:
A parametric human surface or skeleton (SMPL-X, MHR, SOMA-X) whose template points are queried or fitted; not a front end.
_Avoid_: estimator

**Landmark**:
A point on the body template standing in for one anatomical marker, as estimated in a view.
_Avoid_: keypoint (reserve for 2D detections), vertex, marker

**Landmark map**:
The assignment of one template vertex to each marker of the marker set.
_Avoid_: marker map, vertex selection

**Marker set**:
The named anatomical markers the IK fits; the parity set is the 35 markers the RTMPose+LSTM baseline can produce.

**Fusion**:
Combining the per-view metric 3D landmarks of one instant into one set in the world frame.
_Avoid_: triangulation (reserve for 2D-to-3D)

**Triangulation**:
Recovering 3D landmarks from their 2D projections in several views.

## Kinematics

**Biomechanical model**:
The ISB-conventioned multibody model whose joint angles are the output.
_Avoid_: human model, skeleton

**MHE**:
Moving-horizon IK fitting the biomechanical model to the fused landmarks of the last N frames.
_Avoid_: SWIKA, sliding-window IK

**Per-frame IK**:
IK solved independently at each frame.
_Avoid_: SBS, step-by-step

## Evaluation

**Reference**:
Joint angles and markers from the reference system, processed through the same biomechanical model and IK.
_Avoid_: ground truth, mocap, SS

**Arm**:
One pipeline configuration under evaluation (front end × views × fusion × IK).

**Development subset**:
Participants on which any choice (landmark map, filter, horizon) may be tuned; never reported as test results.

**Held-out participants**:
Participants excluded from every selection and tuning step, on which results are reported.

**Fold**:
One of six disjoint groups of three participants; results for a fold come from choices made on the other fifteen only.
_Avoid_: split, partition

**Causal error**:
Error between estimate and reference at the same timestamp, so lag counts as error.
_Avoid_: zero-lag RMSE, unaligned error

**Aligned error**:
Error after shifting the estimate in time to best match the reference; reported only next to the causal error and the lag it removes.

**End-to-end latency**:
Time from image exposure to joint angles available at the output.
_Avoid_: processing rate, throughput, fps
