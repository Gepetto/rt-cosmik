# 14: Calibration inside RT-COSMIK

**What to build:** Human-pose calibration and the checkerboard fallback moved from the separate calibration repository into the toolbox, producing extrinsics in the toolbox's calibration format. The calibration seam is tested with synthetic poses seen by cameras of known extrinsics.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Calibration runs from the toolbox install alone
- [ ] Synthetic tests: rotation and translation error within tolerance, with and without noise
- [ ] Degenerate (flat) recordings rejected
