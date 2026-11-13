# 06: FastSAM-3D out-of-fold arm

**What to build:** FastSAM-3D markers re-exported from the stored full meshes with each fold's map and scored, without re-running inference.

**Blocked by:** 04, 05

**Status:** done (official pass at aba93ac: 108/108 trials in each of the four configurations, 0 solver failures); the runs will be regenerated from the merged branch

- [x] All FastSAM-3D camera configurations scored out-of-fold (`scripts/bash/run_fastsam_oof.sh`; the stored meshes cover camera labels 2, 4, 6, so the configurations are 2, 4-6, 2-4 and 2-4-6, and 3361 runs their mirror image)
- [x] Run information names the fold map used

Joint error in degrees, participant mean ± SD (18 participants), commit aba93ac:

| configuration | views | causal | aligned |
|---|---|---|---|
| `fastsam_2` | 1 camera | 9.71 ± 1.28 | 9.41 |
| `fastsam_4-6` | 2 cameras, same side | 9.26 ± 1.27 | 9.07 |
| `fastsam_2-4` | 2 cameras, opposed | 8.80 ± 1.24 | 8.50 |
| `fastsam_2-4-6` | 3 cameras | 8.77 ± 1.26 | 8.50 |

One trial of `fastsam_4-6`, 1508/Polishing, has an aligned lag of 61 frames (sync r = 0.29: the knee correlation did not lock); only its aligned figure is affected (11.44 against 8.50 causal).
