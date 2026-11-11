# 04: Causal scoring and re-scored sweeps

**What to build:** Per-trial rows report causal error next to aligned error, lag, bias, error SD, marker-space error, jitter, solver failures and constraint violations. The existing horizon, filter, per-frame IK and fusion sweeps are re-scored with these columns.

**Blocked by:** None (can start immediately)

**Status:** code done; sweeps to be re-run on the ticket 07 recordings

- [x] Injected lag on a synthetic trial appears in causal error and is removed by alignment, with the estimated lag equal to the injection (marker-space lag exact; joint-space lag shifted by exactly the injection)
- [ ] Existing sweeps re-scored and tabulated with causal error (re-run of the studies on the out-of-fold recordings)
- [ ] How COMFI synchronizes mocap and video is established; any residual clock offset is estimated from a delay-free signal and removed, so causal error counts only pipeline lag (done in code: as in COMFI's own comparison script, one offset per trial, but measured on the fused landmarks before filter and IK; `measure_clock_offsets.py` writes `config/clock_offsets.json` from the ticket 07 recordings; 2 or 3 frames on the trials measured so far)
- [x] Bias and SD per DoF satisfy RMSE^2 = bias^2 + SD^2
