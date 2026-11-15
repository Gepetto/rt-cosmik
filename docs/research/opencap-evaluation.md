# Evaluation on the OpenCap LabValidation dataset (ticket 16)

State on 2026-11-15. The paper pipeline (NLF-S per view, unweighted mean fusion, tuned 1-euro filter, MHE with N = 7, parity marker set, out-of-fold landmark map) run on the laboratory validation data released with OpenCap (SimTK project `opencap`, release v0.1), against a reference built from the dataset's mocap markers through our model and IK. No COMFI number changes.

Where things are: code in `src/rtcosmik/paper/opencap.py`, `scripts/python/paper/opencap_eval.py` and `opencap_report.py`; data in `/root/workspace/OpenCap/data/LabValidation_withVideos` (read only); everything produced under `/root/workspace/oof/t16` (`views`, `synced`, `calibration`, `runs/<subject>/<trial>/<arm>`, `results`, `logs`, OCP artefacts in `ocp`). Not in git.

## Summary

- **Used**: 9 subjects (the tenth has no videos), 144 dynamic trials (16 per subject), 26 scored DoF. Runs at commit `609abe5`; 0 solver failures.
- **Joint error against our reference, causal, mean ± SD over subjects** (frames of OpenCap's own analysis): five cameras 8.8° ± 1.0°, three cameras 9.3° ± 1.0°, two cameras (±45°) 9.2° ± 1.0°. Aligned error is the same to 0.1° (the pipeline's lag is under half a frame at 60 Hz). On COMFI the same pipeline gives 8.16° ± 0.82° with four cameras.
- **By activity**, five cameras: squat 8.4°, sit-to-stand 8.6°, drop jump 10.4°, walking 7.4°.
- **Marker error** after a rigid alignment per session: 58 ± 17 mm (five cameras), 68 ± 17 mm (two). It grows with the subject's height: NLF's fused landmarks under-size tall subjects (10 % for the 1.96 m subject, 92 mm) and no view corrects it.
- **Forearm pronation** carries a 28° offset on both sides and an RMSE of 30°; the mean over the other 24 DoF is 7.1° (five cameras) and 7.4° (two).
- **OpenCap's own error, recomputed from its files** on the same trials: 5.5° RMSE and 4.7° mean absolute error over its 18 lower-body and trunk rotations with two cameras and HRNet, 5.3° and 4.5° with five; 8.1° to 9.2° RMSE over all 28 rotations, arms included.
- **Matched DoF**: over hip, knee, ankle flexion and lumbar extension and bending, ours 5.9° (five cameras) and 6.2° (two) against OpenCap's 5.5° and 5.7° (HRNet); with shoulder and elbow flexion added, ours 6.6° and 7.0° against 8.0° and 8.4°. Different models, different references, and OpenCap's processing is offline.
- **Two things in the data had to be handled** and change how the results should be read: the released "synced" clips are not synchronous to the frame (section 2.6), and bystanders are visible (section 2.7).

## 1. Data

- 10 subjects; `subject6` has no videos, so 9 are used (`folds.EXTERNAL_PARTICIPANTS`).
- Per subject one static trial and up to 16 dynamic ones: squats (2), sit-to-stand (2), drop jumps (6) in session 0, walking (6) in session 1. A dynamic trial is used when it has mocap markers, OpenCap's mocap joint angles and a video on each of the five cameras: 144 trials, 16 per subject (2 squat, 2 sit-to-stand, 6 drop-jump, 6 walking).
- Videos: five phones at -70°, -45°, 0°, +45°, +70° (Cam0 to Cam4), 60 fps, portrait 720 x 1280. One intrinsic matrix per phone model; extrinsics per session from a checkerboard on the back wall.
- Mocap: 100 Hz, 51 markers (55 in the static trial), no head or hand markers.
- OpenCap's own results are in the dataset: joint angles from mocap (`OpenSimData/Mocap/IK`) and from video for three 2D detectors and 2, 3 or 5 cameras (`OpenSimData/Video/...`), on the mocap clock.

## 2. Method

### 2.1 What is kept from the paper evaluation, and what the frame rate changes

`opencap.dataset_settings` copies the settings with `settings.opencap` applied: 60 Hz (the videos are not resampled), portrait images, clock offset 0. Everything else is `settings.paper`.

| Depends on the frame rate | Handling |
|---|---|
| 1-euro filter | takes `settings.fs`; its cutoffs are in Hz and Hz per m/s, unchanged (2 Hz, beta 15) |
| OCP time step | baked into the acados code: artefacts for 60 Hz are generated in their own folder (`RTCOSMIK_OCP_DIR=/root/workspace/oof/t16/ocp`) |
| MHE horizon | kept at N = 7 frames: 0.10 s here against 0.15 s on COMFI. Not retuned |
| Shoulder re-seed hold | was 20 frames; now 0.5 s (`shoulder_branch.HOLD_SECONDS`), 20 frames at 40 Hz as before, 30 here |
| Lag search of the aligned error | was ±120 frames; bounded here to ±0.5 s (`settings.opencap["max_lag_s"]`), because a walking trial lasts 1.3 s |
| Jitter | reported in degrees per frame, so not comparable with COMFI's |

The detector's TensorRT engine for five views (batch 5, 640 px) takes portrait images through its letterbox; NLF runs in TorchScript, per-view detection. No new engine was built.

### 2.2 World frames

Both arms run in a world with z up, as on COMFI: the estimate in the checkerboard frame turned so that z is up (`BOARD_TO_WORLD`), the reference in the mocap frame turned the same way (`LAB_TO_WORLD`). In the checkerboard's own frame (y down) the calibration IK, which starts from the model's neutral posture, settled in a folded posture on the static trial (hips at 82° and 89°, knees at 48° and 55°), and the marker registration then absorbed it: every joint angle of the estimate carried the offset (22.9° of error on subject4). `opencap.require_standing` now refuses a calibration whose hips or knees are flexed beyond 30°.

### 2.3 Calibration on the static trial, and the start of a trial

- The model is scaled and its markers registered once per subject and arm, on the median posture of the static trial: the static mocap for the reference, the static video (fused landmarks of the arm's own cameras) for the estimate.
- The reference's static posture includes the four medial knee and ankle markers, which exist in the static trial only. They scale the shanks and thighs and build the segment frames; they are not registered on the model and not tracked (`mks_registration(markers=...)`).
- Each dynamic trial then starts with `HumanSolver.start_trial`: the calibration posture is moved onto the first frame's markers and the horizon is converged on that frame with the same solver the calibration uses, five times, each anchored on the previous solution. The causal loop starts from there. Sit-to-stand trials start seated, walking trials mid-stride.

### 2.4 Marker mapping and scored DoF

Our mocap marker set has 32 markers. 25 exist in OpenCap's dynamic trials and 4 more in the static trial:

| Ours | OpenCap | Ours | OpenCap |
|---|---|---|---|
| RASI, LASI | r.ASIS, L.ASIS | RKNE, LKNE | r_knee, L_knee |
| RPSI, LPSI | r.PSIS, L.PSIS | RANK, LANK | r_ankle, L_ankle |
| C7 | C7 | RHEE, LHEE | r_calc, L_calc |
| RSHO, LSHO | R_Shoulder, L_Shoulder | R5MHD, L5MHD | r_5meta, L_5meta |
| RELB, LELB | R_elbow_lat, L_elbow_lat | RTOE, LTOE | r_toe, L_toe |
| RMELB, LMELB | R_elbow_med, L_elbow_med | RMKNE, LMKNE (static only) | r_mknee, L_mknee |
| RWRI, LWRI | R_wrist_radius, L_wrist_radius | RMANK, LMANK (static only) | r_mankle, L_mankle |
| RMWRI, LMWRI | R_wrist_ulna, L_wrist_ulna | Head, REar, LEar | none |

- On COMFI, RWRI is the marker nearer the second metacarpal and RMWRI the one nearer the fifth (86 against 96 mm, and 104 against 69 mm, participant 1012), hence radius and ulna.
- OpenCap's thigh, shank, sternum, humerus and forearm markers and its hip centres have no counterpart in our model and are not used.
- Reference marker set `opencap_mocap` (`rtcosmik.marker_sets`): the 25 markers; locked DoF are the 7 of the parity set plus the 3 cervical ones, which no marker observes.
- **Scored DoF: 26.** The 29 of the paper evaluation minus cervical flexion, lateral bending and rotation, for both arms: hip (3), knee, ankle (2) on each side, lumbar (2), clavicle (1), shoulder (3) and elbow (2) on each side.
- The estimate keeps the paper's parity set (35 landmarks, head included): replaying the five-camera arm without the six head landmarks and with the cervical DoF locked (`opencap_parity`) changes the error on the 26 scored DoF by +0.03° on average (8.68° against 8.65°, whole trials; per-trial difference SD 0.39°, per-subject means from -0.27° to +0.55°). The head hangs off the last joint of the chain, so the paper set is kept unchanged.

### 2.5 Landmark map

No OpenCap subject was used to fit any landmark map, so every fold's map is out of fold for them. `folds.EXTERNAL_PARTICIPANTS` declares the nine subjects and `folds.EXTERNAL_FOLD = 0` gives them fold 0's map; `landmark_maps.require_out_of_fold` now refuses a participant that is neither a COMFI participant nor a declared external one. Each run records the map's fold, the 15 participants it was fitted on, its hash, `in_sample: false` and `external_participant: true`.

### 2.6 Synchronisation

The dataset's `<trial>_syncdWithMocap.avi` clips were to be taken as synchronised. They are not, to the frame:

- Each clip is a run of frames of the camera's whole video starting at a key frame (key frames are 12 frames apart), each camera on its own. Measured by matching the clips' frames in the whole videos: every clip starts at a multiple of 12 frames, at different places in different cameras.
- The cameras of one trial are therefore offset from each other by up to 11 frames (183 ms). Measured on the landmarks: the five cameras of a trial span 8 to 10 frames on average, 11 at most in the drop jumps, up to 14 in squats and sit-to-stand and 16 in walking (section 4.3).
- The clips' time stamps do not say where the mocap trial starts either: in session 0 the first frame of a clip was filmed between 0.18 s before and 0.28 s after the start of the mocap trial (one drop jump 0.50 s after); in the walking trials 0.13 to 0.75 s after it (mean 0.37 s), so a walking clip misses the start of the mocap trial and runs past its end.

What is done instead (stage `sync`, `opencap.camera_offsets`, `align_cameras`, `clock_offset`):

1. NLF is run on each camera's whole video from 60 frames before the clip to 24 frames after it.
2. **Camera offsets.** For each pair of cameras, the whole-frame lag that minimises the distance between their world-frame landmarks, each camera's mean removed (a view's depth bias is not a lag), on the frames of the clip, within ±14 frames. The five offsets are the least-squares fit to the ten pair lags, rounded to whole frames. The rows are shifted so that one row is one instant.
3. **Clock.** The time of the rows on the mocap clock is the offset that minimises the distance, after a rigid fit, between the five-camera landmarks (median over cameras, before any filter or IK) and the 25 mocap markers, searched on a 1/240 s grid. This is COMFI's per-trial clock offset (`measure_clock_offsets.py`), measured the same way, with a finer step because the reference is resampled anyway. The rows inside the mocap trial are kept.
4. The reference is the mocap resampled (linear interpolation) at the times of those rows, so estimate and reference share frame indices and the clock offset at scoring is 0. All arms replay the same synchronised rows.

The offsets are whole frames: the phones are not triggered together, so up to half a frame (8 ms) remains between cameras. The clock uses the reference markers, as on COMFI; the pipeline's own lag (filter, MHE) is not removed by it, since it is measured before the filter.

### 2.7 Person followed in each view

The pipeline follows the person the detector is most confident of on the first frame. Bystanders are visible in the laboratory, and before a clip starts they are alone in some views. The recording therefore picks the person on the clip's first frame, follows them forwards, then backwards through the frames before the clip (`NLFEstimator.tracking_state`, `restore_tracking`). The share of views whose landmarks sit more than 1 m from the cameras' median is reported with the results.

### 2.8 Scoring

- `scoring.score_trial`, as on COMFI, with four options added: columns left unscored (the cervical DoF), a bound on the lag search, a window of frames, and the rigid transform that takes the run's markers into the reference's frame.
- **Causal error**: estimate and reference at the same frame (clock offset 0). **Aligned error**: after the shift that maximises the knee-flexion correlation, within ±0.5 s.
- **Window.** OpenCap's angles cover a part of each drop-jump trial only (about 1 s around the landing) and the whole of the other trials. Scores are given on that window, so that our errors and OpenCap's are on the same frames, and on the whole synchronised trial.
- **Marker error.** The estimate's markers are in the checkerboard world, the reference's in the mocap world. One rigid transform (rotation and translation, no scale) per subject and session is fitted on the five-camera arm's markers against the reference's, the 25 common markers and all the session's trials pooled, and applied to every arm (`opencap_report.session_frames`). The error it leaves is the mean distance over the common markers.
- Statistics: each subject's mean over its trials, then mean ± SD over the 9 subjects.

### 2.9 Comparison with OpenCap's released angles

- **OpenCap's own error** is recomputed from its files: video angles against mocap angles at the same time stamps, RMSE and mean absolute error per coordinate, per trial, on the same 9 subjects and trials.
- **Matched DoF.** Hip (3), knee, ankle flexion, lumbar extension and bending, shoulder (3), elbow flexion. OpenCap's lumbar rotation has no scored counterpart: our trunk rotation is the thoracic one, locked in the parity set. The table sets our error against our reference beside OpenCap's error against its reference. The two use different models, different references (ours has no thigh or shank clusters) and different processing: OpenCap's is offline and non-causal (whole-trial filtering and an LSTM over the trial), ours is causal.

## 3. Results

Mean ± SD over the 9 subjects of each subject's mean over its trials. Tables as written by the `report` stage (`/root/workspace/oof/t16/results/tables.md`; per-trial rows in `trials.csv`, per-DoF rows in `per_dof.csv`).

### 3.1 Arms

On the frames of OpenCap's analysis (37,989 frames):

| arm | subjects | trials | causal deg | aligned deg | causal marker mm | aligned marker mm | joint lag frames | marker lag frames | solver failures | dropped frames | limit violation % | jitter deg/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nlf_5cam | 9 | 144 | 8.8 ± 1.0 | 8.8 ± 0.9 | 57.7 ± 16.5 | 57.7 ± 16.6 | 0.28 | 0.46 | 0 | 0 | 0.00 | 0.621 |
| nlf_3cam | 9 | 144 | 9.3 ± 1.0 | 9.3 ± 1.0 | 70.7 ± 21.1 | 70.8 ± 21.0 | 0.28 | -0.02 | 0 | 0 | 0.00 | 0.659 |
| nlf_2cam | 9 | 144 | 9.2 ± 1.0 | 9.2 ± 1.0 | 68.1 ± 17.4 | 68.0 ± 17.6 | 0.28 | 0.10 | 0 | 7 | 0.00 | 0.709 |

On the whole synchronised trials (43,402 frames):

| arm | subjects | trials | causal deg | aligned deg | causal marker mm | aligned marker mm | joint lag frames | marker lag frames | solver failures | dropped frames | limit violation % | jitter deg/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nlf_5cam | 9 | 144 | 8.7 ± 1.0 | 8.6 ± 1.0 | 59.1 ± 14.7 | 58.9 ± 15.1 | 0.40 | 0.47 | 0 | 0 | 0.00 | 0.484 |
| nlf_3cam | 9 | 144 | 9.1 ± 1.1 | 9.1 ± 1.1 | 71.5 ± 19.7 | 71.6 ± 19.9 | 0.38 | 0.01 | 0 | 0 | 0.00 | 0.520 |
| nlf_2cam | 9 | 144 | 9.0 ± 1.1 | 9.0 ± 1.1 | 69.1 ± 15.6 | 69.1 ± 15.9 | 0.35 | 0.12 | 0 | 7 | 0.00 | 0.568 |

- No solver failure. The two-camera arm has 7 frames in which neither camera saw the subject; the IK is not called on them.
- Causal and aligned errors coincide: the reference is sampled on the video clock (section 2.6) and the filter and the MHE add 0.3 frame of joint lag (5 ms).
- Three cameras (-45°, 0°, +45°) are not better than the two at ±45°. In 6 walking trials the frontal camera follows a bystander and the three-camera arm runs on two cameras (section 4.4); the frontal camera also sees the walk and the jump along its line of sight, which may matter and was not tested.
- Without the two forearm pronation DoF: 7.05° ± 0.80° (five cameras), 7.41° ± 0.87° (two).

### 3.2 Per activity

| arm | error | squat (18 trials) | sit_to_stand (18 trials) | drop_jump (54 trials) | walking (54 trials) |
|---|---|---|---|---|---|
| nlf_5cam | causal | 8.4 ± 1.2 | 8.6 ± 1.0 | 10.4 ± 1.2 | 7.4 ± 1.2 |
| nlf_5cam | aligned | 8.4 ± 1.2 | 8.6 ± 1.0 | 10.4 ± 1.2 | 7.5 ± 1.2 |
| nlf_5cam | marker mm | 52.1 ± 15.2 | 54.3 ± 17.0 | 64.8 ± 16.1 | 53.5 ± 19.0 |
| nlf_3cam | causal | 8.7 ± 1.3 | 9.1 ± 0.9 | 10.9 ± 1.3 | 8.0 ± 1.3 |
| nlf_3cam | aligned | 8.7 ± 1.3 | 9.0 ± 0.9 | 10.8 ± 1.3 | 8.0 ± 1.3 |
| nlf_3cam | marker mm | 64.2 ± 20.7 | 66.1 ± 20.0 | 78.0 ± 21.6 | 67.0 ± 23.5 |
| nlf_2cam | causal | 8.6 ± 1.1 | 9.0 ± 1.0 | 10.6 ± 1.2 | 8.0 ± 1.2 |
| nlf_2cam | aligned | 8.6 ± 1.1 | 9.0 ± 1.0 | 10.6 ± 1.2 | 8.0 ± 1.3 |
| nlf_2cam | marker mm | 60.5 ± 14.6 | 64.6 ± 16.2 | 75.2 ± 17.4 | 64.6 ± 20.8 |

### 3.3 Per DoF

Causal RMSE, aligned RMSE (± SD over subjects) and mean signed error, in degrees.

| DoF | 5cam causal | 5cam aligned | 5cam bias | 3cam causal | 3cam aligned | 3cam bias | 2cam causal | 2cam aligned | 2cam bias |
|---|---|---|---|---|---|---|---|---|---|
| Left_Hip_Flexion_Extension | 8.4 ± 2.3 | 8.4 ± 2.3 | -4.1 | 8.7 ± 2.0 | 8.7 ± 2.0 | -4.0 | 8.5 ± 2.0 | 8.5 ± 2.0 | -4.0 |
| Left_Hip_Abduction_Adduction | 4.6 ± 1.2 | 4.6 ± 1.2 | -3.2 | 5.2 ± 1.3 | 5.2 ± 1.3 | -4.0 | 5.1 ± 1.3 | 5.1 ± 1.2 | -3.8 |
| Left_Hip_Internal_External_Rotation | 5.2 ± 1.2 | 5.2 ± 1.3 | 2.2 | 5.4 ± 1.3 | 5.3 ± 1.4 | 2.9 | 5.0 ± 1.1 | 5.0 ± 1.1 | 2.3 |
| Left_Knee_Flexion_Extension | 4.8 ± 1.7 | 4.6 ± 1.7 | 2.9 | 6.0 ± 1.8 | 5.8 ± 1.8 | 4.0 | 5.5 ± 1.7 | 5.3 ± 1.6 | 3.2 |
| Left_Ankle_Plantarflexion_Dorsiflexion | 7.4 ± 2.3 | 7.3 ± 2.3 | 3.0 | 8.5 ± 2.3 | 8.3 ± 2.3 | 3.2 | 8.3 ± 2.1 | 8.2 ± 2.1 | 2.9 |
| Left_Ankle_Inversion_Eversion | 12.7 ± 3.9 | 12.7 ± 3.9 | -11.1 | 12.4 ± 3.6 | 12.4 ± 3.6 | -10.7 | 12.8 ± 3.6 | 12.8 ± 3.6 | -11.1 |
| Lumbar_Flexion_Extension | 8.0 ± 3.1 | 8.0 ± 3.1 | 6.0 | 8.6 ± 3.4 | 8.6 ± 3.4 | 6.3 | 8.3 ± 3.1 | 8.3 ± 3.1 | 6.0 |
| Lumbar_Lateral_Bending | 3.0 ± 0.6 | 3.0 ± 0.6 | -0.5 | 3.0 ± 0.5 | 3.0 ± 0.5 | -0.6 | 3.1 ± 0.6 | 3.1 ± 0.6 | -0.5 |
| Left_Clavicle_Elevation_Depression | 7.0 ± 1.3 | 7.1 ± 1.3 | 4.7 | 7.3 ± 1.2 | 7.3 ± 1.2 | 4.9 | 7.2 ± 1.3 | 7.2 ± 1.3 | 4.7 |
| Left_Shoulder_Flexion_Extension | 6.7 ± 1.8 | 6.7 ± 1.7 | 0.5 | 7.6 ± 1.9 | 7.7 ± 2.0 | 0.2 | 7.4 ± 1.7 | 7.4 ± 1.8 | 0.1 |
| Left_Shoulder_Abduction_Adduction | 8.4 ± 1.9 | 8.3 ± 1.9 | -5.3 | 8.5 ± 1.7 | 8.5 ± 1.7 | -5.3 | 8.3 ± 1.8 | 8.3 ± 1.7 | -4.7 |
| Left_Shoulder_Internal_External_Rotation | 9.4 ± 2.6 | 9.4 ± 2.6 | 2.2 | 9.4 ± 2.5 | 9.4 ± 2.6 | 2.0 | 9.8 ± 2.7 | 9.8 ± 2.7 | 2.0 |
| Left_Elbow_Flexion_Extension | 6.2 ± 1.3 | 6.2 ± 1.3 | -0.3 | 7.1 ± 1.6 | 7.1 ± 1.6 | -0.3 | 6.8 ± 1.5 | 6.8 ± 1.5 | -0.3 |
| Left_Elbow_Pronation_Supination | 30.4 ± 10.3 | 30.4 ± 10.3 | 28.2 | 29.8 ± 10.0 | 29.8 ± 10.0 | 27.4 | 30.3 ± 10.0 | 30.2 ± 9.9 | 28.0 |
| Right_Clavicle_Elevation_Depression | 6.5 ± 1.4 | 6.5 ± 1.5 | 4.4 | 6.9 ± 1.5 | 6.9 ± 1.5 | 4.7 | 6.9 ± 1.6 | 6.9 ± 1.7 | 4.9 |
| Right_Shoulder_Flexion_Extension | 6.6 ± 1.9 | 6.6 ± 1.9 | 2.1 | 7.5 ± 2.4 | 7.5 ± 2.5 | 2.0 | 7.2 ± 2.2 | 7.2 ± 2.2 | 2.6 |
| Right_Shoulder_Abduction_Adduction | 7.1 ± 1.2 | 7.1 ± 1.2 | -2.9 | 7.0 ± 0.9 | 7.0 ± 0.9 | -2.5 | 7.4 ± 1.0 | 7.4 ± 1.0 | -2.8 |
| Right_Shoulder_Internal_External_Rotation | 12.3 ± 3.8 | 12.3 ± 3.8 | 7.7 | 12.5 ± 3.8 | 12.6 ± 3.9 | 7.8 | 12.9 ± 4.1 | 13.0 ± 4.2 | 8.2 |
| Right_Elbow_Flexion_Extension | 6.5 ± 1.6 | 6.5 ± 1.6 | -3.3 | 7.6 ± 2.0 | 7.6 ± 2.0 | -4.1 | 7.2 ± 1.5 | 7.2 ± 1.6 | -4.1 |
| Right_Elbow_Pronation_Supination | 29.9 ± 5.8 | 29.9 ± 5.8 | 28.1 | 30.8 ± 5.7 | 30.8 ± 5.7 | 29.2 | 30.5 ± 6.0 | 30.5 ± 6.0 | 28.7 |
| Right_Hip_Flexion_Extension | 8.2 ± 1.9 | 8.2 ± 1.9 | -3.9 | 8.3 ± 1.8 | 8.4 ± 1.8 | -3.6 | 8.3 ± 1.6 | 8.3 ± 1.6 | -3.7 |
| Right_Hip_Abduction_Adduction | 3.4 ± 0.6 | 3.4 ± 0.6 | 1.2 | 3.2 ± 0.5 | 3.2 ± 0.5 | 0.6 | 3.4 ± 0.6 | 3.4 ± 0.6 | 0.4 |
| Right_Hip_Internal_External_Rotation | 6.8 ± 1.8 | 6.8 ± 1.8 | 5.6 | 7.4 ± 1.6 | 7.4 ± 1.6 | 6.3 | 7.2 ± 1.6 | 7.1 ± 1.6 | 5.9 |
| Right_Knee_Flexion_Extension | 5.3 ± 1.8 | 5.1 ± 1.8 | 3.6 | 6.8 ± 1.9 | 6.6 ± 1.9 | 5.0 | 6.1 ± 1.8 | 5.9 ± 1.8 | 4.3 |
| Right_Ankle_Plantarflexion_Dorsiflexion | 6.1 ± 1.5 | 6.0 ± 1.4 | 1.7 | 6.9 ± 1.4 | 6.7 ± 1.4 | 1.4 | 6.7 ± 1.3 | 6.6 ± 1.3 | 1.4 |
| Right_Ankle_Inversion_Eversion | 8.7 ± 1.9 | 8.7 ± 1.9 | 4.6 | 9.1 ± 2.1 | 9.1 ± 2.2 | 5.2 | 8.5 ± 2.0 | 8.6 ± 2.0 | 4.3 |

- Forearm pronation: the estimate sits 28° from the reference on both sides. The twist between NLF's elbow and wrist landmarks is nearly constant (correlation with the mocap's below 0.3), so the estimated pronation stays at its calibration value (58° to 76° over subjects). The reference is calibrated at 36° to 53° in 8 of 9 subjects (69° on subject4) and stays near that value. The same holds on COMFI (pronation RMSE 27.9°), where the reference moves around the estimate's value so no constant offset shows. OpenCap's video angles have 16° to 22° of RMSE on pronation against its mocap angles.
- Left ankle inversion carries an 11° offset, the right one 5°.

### 3.4 OpenCap's released angles against its own mocap angles

Recomputed from `OpenSimData/Video/<detector>/<n>-cameras/IK` against `OpenSimData/Mocap/IK`, same 9 subjects and 144 trials, per coordinate, RMSE in degrees. The last rows are means over coordinates, ± SD over subjects.

| coordinate | HRNet 2 cam (144 trials) | HRNet 3 cam (144 trials) | HRNet 5 cam (144 trials) | OpenPose_default 2 cam (144 trials) | OpenPose_default 3 cam (144 trials) | OpenPose_default 5 cam (144 trials) | OpenPose_highAccuracy 2 cam (144 trials) | OpenPose_highAccuracy 3 cam (144 trials) | OpenPose_highAccuracy 5 cam (144 trials) |
|---|---|---|---|---|---|---|---|---|---|
| pelvis_tilt | 7.2 | 6.9 | 6.1 | 6.5 | 6.6 | 6.1 | 6.2 | 5.6 | 5.6 |
| pelvis_list | 2.6 | 2.5 | 2.4 | 2.9 | 2.9 | 2.7 | 2.6 | 2.3 | 2.4 |
| pelvis_rotation | 3.5 | 3.4 | 3.6 | 3.6 | 3.4 | 3.9 | 3.4 | 3.2 | 3.5 |
| lumbar_extension | 9.2 | 8.7 | 8.1 | 8.2 | 8.5 | 8.5 | 7.7 | 7.0 | 7.3 |
| lumbar_bending | 3.9 | 3.8 | 3.8 | 4.5 | 4.4 | 4.5 | 4.1 | 3.6 | 3.7 |
| lumbar_rotation | 5.2 | 5.3 | 5.0 | 5.3 | 5.5 | 5.6 | 5.1 | 5.1 | 4.8 |
| hip_flexion_r | 6.7 | 6.6 | 5.9 | 6.7 | 6.8 | 6.6 | 6.4 | 5.6 | 5.7 |
| hip_adduction_r | 3.2 | 3.0 | 3.1 | 3.8 | 3.7 | 3.6 | 3.3 | 3.1 | 3.2 |
| hip_rotation_r | 4.9 | 4.4 | 4.6 | 5.4 | 4.7 | 4.7 | 4.9 | 4.6 | 4.4 |
| knee_angle_r | 4.7 | 5.1 | 5.0 | 5.7 | 6.2 | 6.3 | 5.4 | 5.4 | 5.3 |
| ankle_angle_r | 6.7 | 7.0 | 7.2 | 7.8 | 8.3 | 7.3 | 7.3 | 7.0 | 6.7 |
| subtalar_angle_r | 6.6 | 6.4 | 6.9 | 7.8 | 7.0 | 6.5 | 7.3 | 7.1 | 6.4 |
| hip_flexion_l | 7.0 | 7.0 | 6.5 | 7.0 | 7.5 | 7.1 | 6.5 | 6.1 | 6.2 |
| hip_adduction_l | 3.2 | 3.2 | 3.3 | 3.7 | 3.7 | 3.5 | 3.3 | 3.1 | 3.1 |
| hip_rotation_l | 4.8 | 4.7 | 4.7 | 4.9 | 5.0 | 4.9 | 5.3 | 4.8 | 5.0 |
| knee_angle_l | 4.7 | 5.1 | 5.0 | 5.8 | 6.6 | 6.1 | 5.6 | 5.7 | 5.4 |
| ankle_angle_l | 7.3 | 7.4 | 7.6 | 8.5 | 8.8 | 7.7 | 7.2 | 7.1 | 6.5 |
| subtalar_angle_l | 6.9 | 6.5 | 6.1 | 6.5 | 6.5 | 6.5 | 6.8 | 6.2 | 6.0 |
| arm_flex_r | 13.0 | 12.2 | 11.1 | 16.4 | 16.4 | 13.7 | 13.9 | 13.6 | 12.0 |
| arm_add_r | 9.9 | 9.8 | 10.2 | 9.9 | 10.2 | 9.8 | 9.2 | 9.5 | 9.2 |
| arm_rot_r | 16.8 | 15.6 | 15.3 | 18.7 | 18.9 | 16.7 | 16.5 | 16.3 | 15.3 |
| elbow_flex_r | 14.3 | 13.5 | 13.4 | 14.0 | 14.1 | 13.0 | 13.7 | 13.1 | 13.3 |
| pro_sup_r | 21.5 | 21.8 | 20.7 | 19.9 | 20.1 | 18.7 | 20.2 | 20.7 | 19.4 |
| arm_flex_l | 11.8 | 11.9 | 11.4 | 14.6 | 13.8 | 12.6 | 13.1 | 12.4 | 12.4 |
| arm_add_l | 8.5 | 8.6 | 8.4 | 9.0 | 8.9 | 8.1 | 8.2 | 8.3 | 8.3 |
| arm_rot_l | 15.5 | 16.3 | 15.2 | 17.1 | 17.1 | 15.8 | 15.6 | 15.9 | 15.1 |
| elbow_flex_l | 14.6 | 14.3 | 14.5 | 14.7 | 14.9 | 14.7 | 14.9 | 14.5 | 14.4 |
| pro_sup_l | 17.4 | 18.0 | 16.7 | 16.4 | 17.0 | 15.5 | 16.4 | 17.1 | 15.8 |
| mean RMSE, 18 lower-body and trunk rotations | 5.5 ± 0.7 | 5.4 ± 0.5 | 5.3 ± 0.6 | 5.8 ± 0.7 | 5.9 ± 0.7 | 5.7 ± 0.8 | 5.5 ± 0.7 | 5.1 ± 0.7 | 5.1 ± 0.7 |
| mean MAE, 18 lower-body and trunk rotations | 4.7 ± 0.7 | 4.6 ± 0.5 | 4.5 ± 0.5 | 4.9 ± 0.7 | 5.0 ± 0.7 | 4.8 ± 0.7 | 4.6 ± 0.6 | 4.4 ± 0.6 | 4.3 ± 0.6 |
| mean RMSE, 10 arm rotations | 14.3 ± 3.3 | 14.2 ± 3.1 | 13.7 ± 2.7 | 15.1 ± 3.5 | 15.1 ± 3.8 | 13.9 ± 3.6 | 14.2 ± 3.7 | 14.2 ± 3.8 | 13.5 ± 3.3 |
| mean RMSE, all 28 | 8.6 ± 1.2 | 8.5 ± 1.1 | 8.3 ± 0.9 | 9.1 ± 1.4 | 9.2 ± 1.5 | 8.6 ± 1.4 | 8.6 ± 1.5 | 8.4 ± 1.6 | 8.1 ± 1.3 |

Per activity, mean RMSE over the 18 lower-body and trunk rotations:

| activity | HRNet 2 cam | HRNet 3 cam | HRNet 5 cam | OpenPose_default 2 cam | OpenPose_default 3 cam | OpenPose_default 5 cam | OpenPose_highAccuracy 2 cam | OpenPose_highAccuracy 3 cam | OpenPose_highAccuracy 5 cam |
|---|---|---|---|---|---|---|---|---|---|
| squat | 4.9 ± 0.8 | 4.8 ± 0.7 | 4.7 ± 0.6 | 5.0 ± 0.9 | 5.0 ± 0.8 | 4.8 ± 1.1 | 4.6 ± 1.0 | 4.5 ± 0.8 | 4.4 ± 0.9 |
| sit_to_stand | 5.5 ± 0.9 | 5.4 ± 0.6 | 5.3 ± 0.9 | 5.5 ± 0.8 | 5.5 ± 0.7 | 5.2 ± 0.8 | 5.1 ± 0.8 | 5.0 ± 0.7 | 4.8 ± 0.8 |
| drop_jump | 6.2 ± 1.0 | 6.0 ± 0.8 | 5.9 ± 0.9 | 6.3 ± 1.0 | 6.3 ± 0.8 | 6.1 ± 0.9 | 5.8 ± 1.0 | 5.6 ± 0.8 | 5.5 ± 0.8 |
| walking | 5.0 ± 0.4 | 4.9 ± 0.5 | 4.9 ± 0.5 | 5.7 ± 0.9 | 6.0 ± 0.9 | 5.6 ± 0.7 | 5.5 ± 1.2 | 5.0 ± 0.7 | 4.9 ± 0.6 |

OpenCap's publication reports a mean absolute error of about 4.5° over these 18 rotations; that figure is quoted from memory and was not checked against the paper. The files give 4.3° to 5.0° depending on detector and cameras.

### 3.5 Matched DoF

Ours against our reference (causal and aligned), OpenCap's against its reference, RMSE in degrees, mean over subjects, left and right pooled.

| DoF | ours 5cam causal | ours 3cam causal | ours 2cam causal | ours 5cam aligned | ours 3cam aligned | ours 2cam aligned | OpenCap HRNet 2 cam | OpenCap HRNet 5 cam | OpenCap OpenPose_default 2 cam | OpenCap OpenPose_default 5 cam |
|---|---|---|---|---|---|---|---|---|---|---|
| lumbar_extension | 8.0 | 8.6 | 8.3 | 8.0 | 8.6 | 8.3 | 9.2 | 8.1 | 8.2 | 8.5 |
| lumbar_bending | 3.0 | 3.0 | 3.1 | 3.0 | 3.0 | 3.1 | 3.9 | 3.8 | 4.5 | 4.5 |
| hip_flexion | 8.3 | 8.5 | 8.4 | 8.3 | 8.5 | 8.4 | 6.8 | 6.2 | 6.8 | 6.8 |
| hip_adduction | 4.0 | 4.2 | 4.3 | 4.0 | 4.2 | 4.3 | 3.2 | 3.2 | 3.7 | 3.6 |
| hip_rotation | 6.0 | 6.4 | 6.1 | 6.0 | 6.3 | 6.0 | 4.9 | 4.6 | 5.1 | 4.8 |
| knee_angle | 5.0 | 6.4 | 5.8 | 4.9 | 6.2 | 5.6 | 4.7 | 5.0 | 5.7 | 6.2 |
| ankle_angle | 6.8 | 7.7 | 7.5 | 6.6 | 7.5 | 7.4 | 7.0 | 7.4 | 8.2 | 7.5 |
| arm_flex | 6.6 | 7.5 | 7.3 | 6.6 | 7.6 | 7.3 | 12.4 | 11.3 | 15.5 | 13.2 |
| arm_add | 7.8 | 7.8 | 7.9 | 7.7 | 7.8 | 7.9 | 9.2 | 9.3 | 9.5 | 9.0 |
| arm_rot | 10.9 | 10.9 | 11.4 | 10.9 | 11.0 | 11.4 | 16.2 | 15.2 | 17.9 | 16.2 |
| elbow_flex | 6.4 | 7.4 | 7.0 | 6.4 | 7.3 | 7.0 | 14.4 | 14.0 | 14.3 | 13.8 |
| mean | 6.6 | 7.1 | 7.0 | 6.6 | 7.1 | 7.0 | 8.4 | 8.0 | 9.0 | 8.5 |
| mean without shoulder and elbow | 5.9 | 6.4 | 6.2 | 5.8 | 6.4 | 6.2 | 5.7 | 5.5 | 6.0 | 6.0 |

- The caveats of section 2.9 apply: two models, two references, and OpenCap filters each trial offline while ours is causal.
- Lower limbs and lumbar (7 rows): OpenCap is lower by 0.4° to 0.5° with HRNet at equal cameras, equal within 0.2° with OpenPose.
- Shoulder and elbow: ours is lower by 2° to 8° per DoF. The shoulder angles of the two models are different descriptions of the arm (section 4.1), so these rows compare each method with its own reference, not the same angle.

## 4. Checks

### 4.1 Our reference against OpenCap's mocap angles

Our reference angles (mocap markers through our model and IK) against OpenCap's mocap angles on the matched DoF, per trial: correlation, and RMSE once the sign convention and the mean difference are removed. 144 trials.

| DoF | median r | RMSE without sign and offset, deg |
|---|---|---|
| lumbar_extension | -0.990 | 1.5 |
| lumbar_bending | 0.933 | 1.3 |
| hip_flexion | 0.999 | 1.5 |
| hip_adduction | -0.918 | 1.8 |
| hip_rotation | 0.856 | 3.4 |
| knee_angle | 0.999 | 1.9 |
| ankle_angle | 0.991 | 2.2 |
| arm_flex | 0.991 | 9.6 |
| arm_add | -0.730 | 7.1 |
| arm_rot | 0.911 | 11.2 |
| elbow_flex | 0.975 | 3.8 |

Hip, knee, ankle and lumbar flexion agree to 1.5° to 2.2°, hip rotation to 3.4°. The shoulder does not: the two models decompose the arm's rotation in different orders (correlation -0.73 on abduction).

### 4.2 Start of a trial

Mean absolute joint error over the scored DoF in the first 0.25 s of the scored frames against the rest, and the distance between the model's markers and the measured ones on the first frame after `start_trial`.

| arm | MAE first frames deg | MAE later deg | start residual mm, mean | start residual mm, max | marker lag frames, mean (range) | views dropped as far % | trials with any | trials with a camera dropped throughout |
|---|---|---|---|---|---|---|---|---|
| nlf_5cam | 7.77 | 7.80 | 12.4 | 26.4 | 0.46 (-1 to 1) | 1.00 | 24 | 6 |
| nlf_3cam | 8.13 | 8.19 | 13.5 | 27.1 | -0.02 (-4 to 1) | 1.59 | 21 | 6 |
| nlf_2cam | 8.03 | 8.11 | 13.6 | 29.9 | 0.10 (-3 to 1) | 0.19 | 13 | 0 |

The first quarter second is not worse than the rest (7.8° against 7.8°, five cameras). By activity, five cameras: sit-to-stand, which starts seated, 8.8° against 7.9°; squat 6.6° against 7.7°; drop jump 8.8° against 9.0°; walking 6.8° against 6.7°. For the reference the same residual is 2 to 9 mm (subject4).

### 4.3 Synchronisation

| activity | trials | camera offset span frames, mean | max | clock offset from the clip's nominal start s, mean | range | residual after rigid fit mm | scored share of the synchronised frames |
|---|---|---|---|---|---|---|---|
| drop_jump | 54 | 8.5 | 11 | -0.018 | -0.154 to +0.500 | 63 | 0.38 |
| sit_to_stand | 18 | 8.3 | 14 | +0.002 | -0.179 to +0.112 | 53 | 1.00 |
| squat | 18 | 9.3 | 13 | -0.045 | -0.179 to +0.283 | 51 | 1.00 |
| walking | 54 | 10.1 | 16 | +0.370 | +0.133 to +0.746 | 54 | 1.00 |

- "Camera offset span": the largest minus the smallest of the five whole-frame offsets of a trial. Offsets from body-centred landmarks and from absolute positions agree to one frame; the ten pair lags of a trial agree with the five offsets to 0.08 frame on average, 2 at most (subjects 2, 4 and 10).
- "Clock offset": mocap time at which the clip's first frame was filmed.
- "Residual": mean distance between the five-camera landmarks and the mocap markers at the chosen clock, after a rigid fit per trial. It ranges from 42 mm (subjects 4 and 7) to 89 mm (subject2), following the size error of section 4.5.
- Three sit-to-stand videos end before the mocap trial does (42, 51 and 97 frames short); those frames have no estimate and no reference.
- Before synchronisation, on subject4 with the clips taken as given: 12.6° causal and 154 mm with five cameras (244 mm in walking), against 7.5° and 44 mm after.

### 4.4 Bystanders

The last three columns of the table of section 4.2: share of views replayed as "saw nobody" because their landmarks sat more than 1 m from the cameras' median (`settings.opencap["far_view_m"]`), and the trials concerned. In 6 walking trials (subjects 2, 5, 7 and 8) the frontal camera follows a bystander from the first to the last frame; the five-camera arm then runs on four cameras and the three-camera arm on two. The two-camera arm has no such trial. The five recorded cameras decide which views are far, also for the two- and three-camera arms.

Without this rule, on those 6 trials and five cameras: 8.8° against 7.9°, and 310 mm of marker error against 71 mm. The 8 other trials of these four subjects with dropped views are unchanged to 0.1° and 0.1 mm (the views dropped in them held nobody). The five other subjects were not replayed without the rule. Runs in `/root/workspace/oof/t16/variant_asis`.

### 4.5 Size of the estimated body

Marker error per subject, five cameras, with the scale that would best match the estimate's markers to the reference's after the rigid fit (session 0; above 1 the estimate is too small):

| Subject | Height m | Marker error mm | Scale | Marker error with that scale mm | Joint error deg |
|---|---|---|---|---|---|
| subject10 | 1.60 | 44 | 0.995 | 44 | 8.3 |
| subject8 | 1.64 | 51 | 0.960 | 47 | 9.1 |
| subject4 | 1.68 | 45 | 1.009 | 44 | 7.5 |
| subject7 | 1.68 | 47 | 1.005 | 46 | 7.9 |
| subject3 | 1.69 | 52 | 1.013 | 51 | 10.1 |
| subject11 | 1.84 | 65 | 1.049 | 59 | 9.4 |
| subject5 | 1.85 | 52 | 1.050 | 42 | 8.7 |
| subject9 | 1.85 | 64 | 1.068 | 49 | 8.2 |
| subject2 | 1.96 | 90 | 1.100 | 66 | 10.2 |

Each view regresses a metric body, and its size is pulled towards an average one; the mean of the views keeps that error, where a triangulation would not. The joint error does not follow it, because the model is scaled on the estimate's own static posture. (Marker errors and scale from one fit over the trials of session 0, whole trials, `/root/workspace/oof/t16/scratch/scale.py`, output in `results/scale_check.txt`; joint error from `trials.csv`.)

### 4.6 Head landmarks in the estimate

Section 2.4: +0.03° without them, the paper set is kept. Runs in `/root/workspace/oof/t16/variant_nohead`.

## 5. Limits

- The synchronisation is ours, not the dataset's: camera offsets to the frame, clock from the reference markers (section 2.6). A dataset with hardware-synchronised cameras would not need it.
- Lens distortion is ignored, as on COMFI (NLF takes the intrinsic matrix only). The phones' distortion moves a point by about 1 % of its distance to the image centre in the area the person occupies.
- The reference lacks head markers and, in dynamic trials, medial knee and ankle markers: the cervical DoF are not scored, and hip rotation in the reference rests on the lateral knee and ankle markers alone.
- The model calibration uses the static trial, which COMFI does not have; on COMFI the first frame of each trial plays that role.
- N = 7 frames at 60 Hz is a shorter horizon in time than on COMFI; filter and horizon were not retuned for 60 Hz or for these activities.
- Drop jumps are scored on about 1 s per trial, walking on about 1.3 s.
- The shoulder angles of the two models do not describe the arm the same way (section 4, reference agreement), so the shoulder rows of the matched table compare errors, not the same quantities.

## 6. Reproducing

```
export RTCOSMIK_OCP_DIR=/root/workspace/oof/t16/ocp OMP_NUM_THREADS=1 MPLBACKEND=Agg
D=/root/workspace/OpenCap/data/LabValidation_withVideos; O=/root/workspace/oof/t16
for stage in record sync reference replay report; do
    python3 scripts/python/paper/opencap_eval.py $stage --dataset $D --out $O
done
```

`record` needs the GPU (about 10 minutes per subject); the other stages run on the CPU. The first `reference` and `replay` jobs build their acados solver alone before the others start. Tables are written to `$O/results/tables.md`.
