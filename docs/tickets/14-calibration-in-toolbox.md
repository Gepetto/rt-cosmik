# 14: Calibration inside RT-COSMIK

**What to build:** Human-pose calibration and the checkerboard fallback moved from the separate calibration repository into the toolbox, producing extrinsics in the toolbox's calibration format. The calibration seam is tested with synthetic poses seen by cameras of known extrinsics.

**Blocked by:** None (can start immediately)

**Status:** done (2026-11-15): `rtcosmik.calibration`, `scripts/python/calibration/`. The human-pose method is a rigid registration of the per-view 3D landmarks; the degenerate recording it rejects is a person who stays in place (findings 0f).

- [x] Calibration runs from the toolbox install alone
- [x] Synthetic tests: rotation and translation error within tolerance, with and without noise
- [x] Degenerate (flat) recordings rejected
