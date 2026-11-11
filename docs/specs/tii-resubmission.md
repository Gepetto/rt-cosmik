# Spec: RT-COSMIK resubmission to IEEE TII (TII-26-9909)

Status: ready-for-agent · Deadline: resubmission by mid-January 2027 · Branch: `tii-resubmission`

## Problem Statement

The RT-COSMIK article was rejected-and-resubmit by IEEE Transactions on Industrial Informatics on 2026-11-11. Reviewers and the Associate Editor accept the relevance of the work but do not accept its evidence. The reported accuracy may be inflated, because the landmark maps of NLF and FastSAM-3D were fitted against the reference of the same participants the results are reported on. The joint-angle errors are aligned errors: a knee-flexion cross-correlation removed the lag of the filter and of the pipeline, so real-time accuracy and end-to-end latency are unreported. The choices of horizon, filter and fusion weights are not justified by any sensitivity data, although most of that data already exists unpublished. The calibration used in the evaluation is the reference calibration, not the toolbox's own deployment workflow, and the human-pose calibration result in the paper is claimed without a protocol. Generalization rests on one dataset, the "low-cost" claim is not quantified, and the ergonomic and human-robot distance claims are supported by mean errors only.

On top of that, the toolbox itself is hard to use: calibration lives in a separate repository, dependencies are undeclared, several front ends and study scripts sit next to the realtime pipeline, and the code that produced the paper was never merged into `main`.

The author cannot record new data (no participants, no live camera rig) and has about two months.

## Solution

Answer every reviewer comment with evidence that comes from replays of existing recordings (COMFI videos and outputs, the OpenCap public validation dataset) and from timing replays on three GPU classes, and simplify the toolbox only where it serves the resubmission.

Concretely: re-fit both landmark maps per fold so every participant is evaluated out-of-fold; make causal error the headline metric with aligned error and lag beside it; make MHE without a low-pass filter the headline arm if it gives the lowest causal error, with a jitter-capped 1€ filter as an optional arm; publish the existing horizon, filter and fusion sweeps as sensitivity analyses; validate the fusion (σ calibration, robust baseline, unreliable views); validate the human-pose calibration against the reference calibration on COMFI and against checkerboard calibration on the OpenCap dataset; measure per-stage latency and deadline misses in sustained replay on an RTX 3050 laptop, an RTX 4070 desktop and the RTX 4500 Ada workstation; strengthen the REBA and robot-distance analyses or narrow their claims; and ship a toolbox with one front end (NLF), one install, calibration included, and one calibrate → run → evaluate path.

The author then writes a point-by-point response to reviewers in which each answer cites the result that supports it.

## User Stories

### Evaluation protocol

1. As the author, I want the 18 COMFI participants divided into six fixed folds of three, so that every result can be reported out-of-fold while keeping n = 18 for paired tests.
2. As the author, I want the fold assignment stored once and versioned with the code, so that every arm and every collaborator uses the same folds.
3. As the author, I want the NLF landmark map fitted per fold on the other fifteen participants, so that no participant's reference influences the landmarks it is evaluated with.
4. As the author, I want the FastSAM-3D landmark map re-fitted per fold from the stored full meshes, so that FastSAM-3D is evaluated cleanly without re-running its inference.
5. As the author, I want the evaluation to refuse to score a participant with a landmark map whose fitting set contains that participant, so that leakage cannot reappear by mistake.
6. As the author, I want the hand-picked landmarks (pelvis, hands, face) documented as fixed a priori, so that reviewers see they were not tuned on the reference.
7. As the author, I want one NLF inference pass over all 108 trials that uses each participant's fold map and records every view, so that all camera configurations and fusion variants can be replayed on CPU afterwards.
8. As the author, I want every run to write its commit, settings, fold, landmark map identity, hardware and date to its run information, so that every number in the paper traces to a reproducible run.
9. As the author, I want runs to refuse to start from a working tree with uncommitted changes to pipeline code, so that the previous untraceable-results situation cannot repeat.
10. As the author, I want the participant used for each figure and the folds listed in the paper, so that the protocol is fully stated.

### Metrics

11. As the author, I want causal error reported for every arm, so that the headline accuracy is what a real-time user experiences.
12. As the author, I want the aligned error and the estimated lag reported next to the causal error, so that reviewers can see what alignment hid.
13. As a reviewer, I want joint-angle bias (absolute offset) and the standard deviation of the error reported per degree of freedom, so that systematic offsets are not hidden by neutral referencing.
14. As a reviewer, I want marker-space errors reported before the IK, so that agreement is not explained only by the estimate and the reference sharing the biomechanical model.
15. As the author, I want jitter reported for every arm, so that the smoothness cost of removing the filter is visible.
16. As the author, I want solver failures and constraint violations counted per arm, so that I can report them as R3.6 asks.

### IK and filtering

17. As the author, I want MHE without any low-pass filter evaluated as a candidate headline arm, so that the paper headlines the lowest causal error.
18. As the author, I want a 1€ filter applied to the fused landmarks as an optional arm, so that users who need smoother output have a low-lag option.
19. As the author, I want the 1€ parameters tuned per fold to minimize causal error with jitter at or below that of the 5 Hz Butterworth, so that the filter claim is "same smoothness, less lag" without tuning on the evaluated participants.
20. As the author, I want the 5 Hz Butterworth kept as a comparison arm, so that the change from the submitted version is quantified.
21. As the author, I want per-frame IK evaluated with exactly the same inputs and filtering as the headline MHE arm, so that the benefit of the MHE is separated from smoothing.
22. As the author, I want the existing horizon sweep (N from 3 to 20) and filter sweep (cutoffs 3–10 Hz, order 2 and 4, none) re-reported with causal error, so that N = 7 and the filter choice are justified as sensitivity analyses.
23. As the author, I want the MHE weights and the scaling of position, orientation, joint-angle and acceleration terms stated, and the reference state indexing of the MHE cost clarified, so that the method is reproducible.

### Fusion

24. As a reviewer, I want inverse-variance fusion compared with the plain mean, reprojection fusion and a per-landmark weighted median under identical conditions, so that the contribution of uncertainty weighting is isolated.
25. As a reviewer, I want to know whether NLF's predicted σ tracks its actual error against the reference, so that inverse-variance weighting is justified.
26. As the author, I want a reliability analysis of σ per landmark and per view, so that I can show where σ is and is not informative.
27. As a reviewer, I want fusion evaluated with views randomly masked, so that robustness to missing views is measured.
28. As a reviewer, I want fusion evaluated with synthetic outliers injected into one view, so that robustness to an unreliable view is measured.
29. As a reviewer, I want the handling of missing landmarks, extreme weights and outliers stated, so that the fusion is fully specified.
30. As the author, I want the occluded robot tasks reported separately in the fusion analyses, so that the occlusion behaviour is visible.

### Calibration

31. As a toolbox user, I want to calibrate the extrinsics of my views from a person walking in the workcell, so that I need no checkerboard or wand.
32. As the author, I want the human-pose calibration compared with the reference calibration on COMFI in rotation and translation error, so that its geometric accuracy is stated with a protocol.
33. As the author, I want the full pipeline replayed with the human-pose calibration and its joint-angle, marker and robot-distance errors compared with the same pipeline under the reference calibration, so that the downstream cost of deployment calibration is known.
34. As the author, I want human-pose calibration compared with checkerboard calibration on the OpenCap dataset, so that the paper discusses the toolbox's practical calibration workflows against marker-based reference kinematics.
35. As the author, I want the sensitivity of joint angles and positions to perturbed extrinsics measured, so that R3.4's sensitivity question is answered.
36. As the author, I want a decision point after the downstream cost is known, so that a calibration method change is only attempted if the cost is large.
37. As a toolbox user, I want calibration included in RT-COSMIK itself, so that I install one package.

### Latency and hardware

38. As a reviewer, I want a per-stage latency breakdown (decode, detection, NLF, fusion, filter, IK, output), so that end-to-end processing time is characterized beyond throughput.
39. As a reviewer, I want latency medians, upper percentiles and maxima during sustained replay, so that variability is reported.
40. As a reviewer, I want deadline misses at 40 Hz counted during sustained replay, so that real-time operation is demonstrated.
41. As the author, I want the same timing replay run on an RTX 3050 laptop, an RTX 4070 desktop and the RTX 4500 Ada workstation, so that the "low-cost" claim is substantiated or dropped.
42. As the author, I want that timing replay to need only a clone of the repository, the documented install and a selected subset of COMFI, so that I can run it myself on each machine.
43. As the author, I want a hardware cost table, so that "low-cost" is quantified.
44. As the author, I want acquisition-side synchronization error described as a limitation with the software synchronization mechanism explained, so that the paper does not claim what replay cannot measure.
45. As the author, I want the camera count to be limited only by built detector engines, documented as such, so that the "one to four cameras" wording is accurate.

### Generalization and comparisons

46. As the author, I want the headline arm run on the OpenCap validation data with the ±45° camera pair and with all five cameras, so that generalization to another laboratory and camera layout is shown.
47. As the author, I want OpenCap's own released results compared with RT-COSMIK on the same trials, so that R2.6's request for an OpenCap comparison is answered.
48. As the author, I want the OpenCap run limited to participants whose videos are released, so that the dataset terms are respected.
49. As the author, I want COMETH compared in the text (inputs, sensors, IK formulation, observer, reported metrics), so that R1's novelty concern is addressed without an infeasible RGB-D run.
50. As the author, I want the novelty argued against per-view skeleton fusion and moving-horizon tracking prior work as the combination plus evaluation against ISB joint angles, so that Eq. (1) is presented honestly as standard inverse-variance averaging.

### Applications

51. As a reviewer, I want REBA called posture-only REBA consistently, so that it is not confused with complete REBA.
52. As a reviewer, I want REBA computed from absolute angles alongside neutral-relative angles, so that the effect of offsets on scoring is visible.
53. As a reviewer, I want risk-level confusion matrices, so that misclassification is visible.
54. As a reviewer, I want representative postures selected on the estimated trajectories, so that the posture-scoring workflow is evaluated as it would be deployed.
55. As a reviewer, I want the underestimation of high-risk postures quantified, so that the ergonomic risk of the errors is known.
56. As a reviewer, I want signed hand-to-end-effector distance errors, their 95th and 99th percentiles, and missed close approaches at stated thresholds computed with causal error, so that distance errors relevant to HRC are characterized.
57. As the author, I want the robot application worded as distance estimation, and dropped if the statistics do not support it, so that no safety claim is made.

### Statistics

58. As a reviewer, I want Holm correction applied within each family of tests, including FastSAM-3D comparisons, so that multiple testing is controlled.
59. As a reviewer, I want equivalence claims made with a two one-sided tests procedure against a ±1° margin stated in advance, so that non-significance is not presented as equivalence.

### Toolbox

60. As a toolbox user, I want every runtime dependency declared, so that one install works.
61. As a toolbox user, I want NLF as the only front end in the toolbox, so that the realtime pipeline is simple.
62. As the author, I want RTMPose+LSTM, FastSAM-3D and the study tooling kept in a clearly separated paper area, so that the comparisons stay reproducible without cluttering the toolbox.
63. As a toolbox user, I want one documented calibrate → run → evaluate path, so that I can deploy without reading the paper scripts.
64. As a toolbox user, I want arm and pipeline choices in the settings file and only paths on the command line, so that configuration stays in one place.
65. As a collaborator, I want the shared vocabulary in the glossary, so that the paper, the code and the response use the same terms.

### Paper and response

66. As a collaborator, I want a response document listing every reviewer comment with its answer or planned action, so that I can see and discuss the plan.
67. As the author, I want each answer in the response to cite the ticket or result that supports it, so that nothing is claimed without evidence.
68. As the author, I want the minor issues fixed (keywords, Fig. 4 labels, missing references [1]–[13], arXiv references replaced by published versions where they exist, the 1196.9 mm standard deviation of NLF-2D with two facing cameras explained), so that R2's minor concerns are closed.
69. As the author, I want the contributions to state the causal pipeline with its latency budget, the 3D fusion versus 2D comparison, the validated human-pose calibration, and deployment guidelines across two datasets, with the leakage-free protocol described in the technical section rather than claimed as a contribution.
70. As the author, I want participant names absent from the repository, the paper and any released file, so that participant privacy is protected.

## Implementation Decisions

- The work branches from the paper branch at the commit whose pipeline code equals the submitted results, not from `main` (ADR 0001). The `main` line has since gained documentation and Docker work; merging the resubmission back is a later task, outside this spec.
- A fold module owns the fixed assignment of participants to six folds and answers "which participants may this fold's choices be fitted on". Every fitting step (both landmark maps, 1€ parameters) takes a fold and receives only the other fifteen participants.
- The existing landmark-map fitting procedure (candidate vertices near anatomical anchors, per-frame similarity alignment, median aggregation, linear assignment) is kept unchanged and applied per fold, for NLF and FastSAM-3D alike. The hand-picked pelvis, hands and face stay fixed for both.
- The arm runner (the existing sweep and view replay) is the single entry point for every evaluated arm. An arm is described by its front end, views, fusion, filter, IK and fold; arm descriptions live in settings, and only paths are given on the command line.
- The arm runner refuses to run when the landmark map for the trial's participant was fitted on a set containing that participant, and when pipeline code has uncommitted changes.
- The per-trial scoring module computes causal error and aligned error side by side, plus lag, bias, error standard deviation, marker-space error, jitter, solver failures and constraint violations. Aggregation, statistics and tables stay downstream of per-trial rows, as today.
- One NLF inference pass over all trials, each participant with its fold map, records all views; every other NLF arm is a CPU replay of that recording.
- The 1€ filter is a drop-in alternative to the current IIR on the fused landmarks, causal and fed one frame at a time like the current filter. Its tuning objective is causal error under a jitter cap equal to the 5 Hz Butterworth's jitter.
- The headline arm is chosen by a rule stated before the out-of-fold results are computed: MHE without filter unless the jitter-capped 1€ arm has lower causal error.
- Fusion gains a per-landmark weighted median. View masking and outlier injection are replay perturbations, not pipeline features.
- The σ reliability analysis compares per-view, per-landmark σ with the actual error against the reference, from the recorded views.
- The human-pose calibration code moves from the separate calibration repository into RT-COSMIK as a calibration module with the same inputs (per-view landmarks of a recording) and outputs (extrinsics in the toolbox's calibration format). The checkerboard path moves with it as a fallback. The reference calibration is used only as ground truth.
- The OpenCap dataset gets a loader producing the same trial layout the arm runner already consumes (views, calibration, reference), and a reference built by driving its markers through the same biomechanical model and IK, mapping its marker set onto ours.
- Timing is a sustained-replay mode of the realtime path that records per-stage timestamps and deadline misses; it runs from a clean clone plus a documented COMFI subset.
- Statistics: Holm within each family; two one-sided tests with ±1° margin for equivalence.
- Output of new runs goes outside the read-only COMFI mount, to a location with enough free space; the container disk is currently almost full.

## Testing Decisions

- Tests check external behaviour at two seams only, never internal functions:
  - **Arm evaluation seam:** recorded per-view landmarks of a trial + arm description + reference → run folder, run information and per-trial metrics row. Tests use synthetic trials built from reference landmarks with injected noise, lag, masked views and outliers. They assert, for example: injected lag appears in causal error and is removed by alignment, with the estimated lag matching the injection; inverse-variance fusion beats the plain mean when σ is truthful and does not when σ is uninformative; the weighted median resists an outlier view; masking a view degrades error but does not crash; a participant scored with a map fitted on its own fold is refused; run information contains commit, fold and map identity.
  - **Human-pose calibration seam:** per-view landmarks of a recording → extrinsics. Tests use synthetic poses observed by cameras with known extrinsics and assert rotation and translation error within tolerance, with and without per-view noise, and rejection of degenerate (flat) recordings.
- Timing and latency are measurements run by the author, not tests.
- Prior art: the unit tests for triangulation, IK and camera handling in the repository's unit test tree; the calibration repository's workflow tests; the replay-versus-direct consistency check documented in the view replay tool (sub-millimetre median marker difference), which becomes a regression test at the arm evaluation seam.

## Out of Scope

- Any new recording: no participants, no live camera rig, no glass-to-glass latency measurement.
- Track B: a learned end-to-end multi-view 3D pose model (training on CMU Panoptic, TotalCapture, Human3.6M, COMFI; random view masking; calibration conditioning). Parked until after the resubmission.
- Running COMETH, OpenCap's pipeline, Pose2Sim or OpenSimRT ourselves; comparisons are textual (COMETH) or use released results (OpenCap).
- Human3.6M as a validation dataset.
- SOMA-X regularization of NLF.
- A new human-pose calibration method, unless the downstream-cost decision point (story 36) calls for it.
- Body-segment-to-robot-link distances and any safety rating.
- Foot-contact work, which continues on its own branch.
- Merging back into `main`, the documentation book and the Docker setup.
- A deeper refactor of the toolbox beyond the simplifications listed.

## Further Notes

- Reviewer mapping: leakage (R1, R2.5, R3.1, AE) → stories 1–10; latency (R1, R2.7, R3.3) → 11–12, 38–45; ablations and sensitivity (R1, R2.2, R3.5, R3.6) → 17–30; calibration (R2.4, R2.13, R3.4) → 31–37; generalization and comparisons (R1, R2.1, R2.3, R2.6) → 46–50; shared model (R3.2) → 13–14; REBA (R2.9, R3.7) → 51–55; robot distance (R2.10, R3.8) → 56–57; statistics (R2.8) → 58–59; low cost (R2.11, R3.4) → 41–43; minor (R2.11–R2.15) → 68.
- Existing results already answer part of the sensitivity questions: on all 18 participants, horizon N barely changes accuracy, the 5 Hz filter costs about 0.5° of causal error and about 80 ms of lag, MHE beats per-frame IK by about 1.4° with matched filtering, and no solver failed. They must be re-reported with causal error, and their fold-dependent arms recomputed.
- The participant-to-name mapping found in the COMFI FastSAM outputs must never be copied into the repository or the paper.
- The author runs the experiments; agents prepare code and commands.
- Vocabulary follows the glossary; decisions follow ADR 0001.
