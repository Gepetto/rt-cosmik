# 19: Sustained-replay timing

**What to build:** A sustained-replay mode recording per-stage timestamps (decode, detection, NLF, fusion, filter, IK, output) and missed 40 Hz deadlines, runnable from main's Docker image plus a documented COMFI subset. The author runs it on the RTX 3050 laptop, the RTX 4070 desktop and the workstation.

**Blocked by:** 18

**Status:** ready-for-agent

- [ ] Per-stage medians, percentiles and maxima
- [ ] Deadline misses counted
- [ ] Procedure needs only the Docker image, a clone and the COMFI subset
- [ ] Dropped frames in COMFI recordings reported from timestamps where available
