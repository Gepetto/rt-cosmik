# 04: Causal scoring and re-scored sweeps

**What to build:** Per-trial rows report causal error next to aligned error, lag, bias, error SD, marker-space error, jitter, solver failures and constraint violations. The existing horizon, filter, per-frame IK and fusion sweeps are re-scored with these columns.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Injected lag on a synthetic trial appears in causal error and is removed by alignment, with the estimated lag equal to the injection
- [ ] Existing sweeps re-scored and tabulated with causal error
- [ ] How COMFI synchronizes mocap and video is established; any residual clock offset is estimated from a delay-free signal and removed, so causal error counts only pipeline lag
- [ ] Bias and SD per DoF satisfy RMSE^2 = bias^2 + SD^2
