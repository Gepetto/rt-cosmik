# Evaluation on the OpenCap LabValidation dataset (ticket 16)

State on 2026-11-15, branch `ticket-16`. The paper pipeline (NLF-S per view, unweighted mean fusion, tuned 1-euro filter, MHE with N = 7, parity marker set, out-of-fold landmark map) run on the laboratory validation data released with OpenCap (SimTK project `opencap`, release v0.1), against a reference built from the dataset's mocap markers through our model and IK. No COMFI number changes.

Where things are: code in `src/rtcosmik/paper/opencap.py`, `scripts/python/paper/opencap_eval.py` and `opencap_report.py`; data in `/root/workspace/OpenCap/data/LabValidation_withVideos` (read only); everything produced under `/root/workspace/oof/t16` (`views`, `synced`, `calibration`, `runs/<subject>/<trial>/<arm>`, `results`, `logs`, OCP artefacts in `ocp`). Not in git.

@@SUMMARY@@

## 1. Data

- 10 subjects; `subject6` has no videos, so 9 are used (`folds.EXTERNAL_PARTICIPANTS`).
- Per subject one static trial and up to 16 dynamic ones: squats (2), sit-to-stand (2), drop jumps (6) in session 0, walking (6) in session 1. A dynamic trial is used when it has mocap markers, OpenCap's mocap joint angles and a video on each of the five cameras: @@NTRIALS@@.
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
- The estimate keeps the paper's parity set (35 landmarks, head included): @@HEAD@@

### 2.5 Landmark map

No OpenCap subject was used to fit any landmark map, so every fold's map is out of fold for them. `folds.EXTERNAL_PARTICIPANTS` declares the nine subjects and `folds.EXTERNAL_FOLD = 0` gives them fold 0's map; `landmark_maps.require_out_of_fold` now refuses a participant that is neither a COMFI participant nor a declared external one. Each run records the map's fold, the 15 participants it was fitted on, its hash, `in_sample: false` and `external_participant: true`.

### 2.6 Synchronisation

The dataset's `<trial>_syncdWithMocap.avi` clips were to be taken as synchronised. They are not, to the frame:

- Each clip is a run of frames of the camera's whole video starting at a key frame (key frames are 12 frames apart), each camera on its own. Measured by matching the clips' frames in the whole videos: every clip starts at a multiple of 12 frames, at different places in different cameras.
- The cameras of one trial are therefore offset from each other by up to 11 frames (183 ms). Measured on the landmarks: @@CAMOFFSETS@@
- The clips' time stamps do not say where the mocap trial starts either: @@CLOCK@@

What is done instead (stage `sync`, `opencap.camera_offsets`, `align_cameras`, `clock_offset`):

1. NLF is run on each camera's whole video from 60 frames before the clip to 24 frames after it.
2. **Camera offsets.** For each pair of cameras, the whole-frame lag that minimises the distance between their world-frame landmarks, each camera's mean removed (a view's depth bias is not a lag), on the frames of the clip, within ±14 frames. The five offsets are the least-squares fit to the ten pair lags, rounded to whole frames. The rows are shifted so that one row is one instant.
3. **Clock.** The time of the rows on the mocap clock is the offset that minimises the distance, after a rigid fit, between the five-camera landmarks (median over cameras, before any filter or IK) and the 25 mocap markers, searched on a 1/240 s grid. This is COMFI's per-trial clock offset (`measure_clock_offsets.py`), measured the same way, with a finer step because the reference is resampled anyway. The rows inside the mocap trial are kept.
4. The reference is the mocap resampled (linear interpolation) at the times of those rows, so estimate and reference share frame indices and the clock offset at scoring is 0. All arms replay the same synchronised rows.

The offsets are whole frames: the phones are not triggered together, so up to half a frame (8 ms) remains between cameras. The clock uses the reference markers, as on COMFI; the pipeline's own lag (filter, MHE) is not removed by it, since it is measured before the filter.

### 2.7 Person followed in each view

The pipeline follows the person the detector is most confident of on the first frame. Bystanders are visible in the laboratory, and before a clip starts they are alone in some views. The recording therefore picks the person on the clip's first frame, follows them forwards, then backwards through the frames before the clip (`NLFEstimator.tracking_state`, `restore_tracking`). The share of views whose landmarks sit more than 1 m from the cameras' median is reported with the results.

### 2.8 Scoring

- `scoring.score_trial`, as on COMFI, with three options added: columns left unscored (the cervical DoF), a bound on the lag search, a window of frames, and the rigid transform that takes the run's markers into the reference's frame.
- **Causal error**: estimate and reference at the same frame (clock offset 0). **Aligned error**: after the shift that maximises the knee-flexion correlation, within ±0.5 s.
- **Window.** OpenCap's angles cover a part of each drop-jump trial only (about 1 s around the landing) and the whole of the other trials. Scores are given on that window, so that our errors and OpenCap's are on the same frames, and on the whole synchronised trial.
- **Marker error.** The estimate's markers are in the checkerboard world, the reference's in the mocap world. One rigid transform (rotation and translation, no scale) per subject and session is fitted on the five-camera arm's markers against the reference's, the 25 common markers and all the session's trials pooled, and applied to every arm (`opencap_report.session_frames`). The error it leaves is the mean distance over the common markers.
- Statistics: each subject's mean over its trials, then mean ± SD over the 9 subjects.

### 2.9 Comparison with OpenCap's released angles

- **OpenCap's own error** is recomputed from its files: video angles against mocap angles at the same time stamps, RMSE and mean absolute error per coordinate, per trial, on the same 9 subjects and trials.
- **Matched DoF.** Hip (3), knee, ankle flexion, lumbar extension and bending, shoulder (3), elbow flexion. OpenCap's lumbar rotation has no scored counterpart: our trunk rotation is the thoracic one, locked in the parity set. The table sets our error against our reference beside OpenCap's error against its reference. The two use different models, different references (ours has no thigh or shank clusters) and different processing: OpenCap's is offline and non-causal (whole-trial filtering and an LSTM over the trial), ours is causal.

@@RESULTS@@

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
