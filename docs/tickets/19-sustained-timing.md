# 19: Sustained-replay timing

**What to build:** A sustained-replay mode recording per-stage timestamps (decode, detection, NLF, fusion, filter, IK, output) and missed 40 Hz deadlines, runnable from main's Docker image plus a documented COMFI subset. It is developed and first measured on the workstation GPU. The RTX 3050 Ti laptop and the RTX 4070 desktop are measured by the author at the very end, from a clean install.

**Blocked by:** 18, 22

**Status:** ready-for-agent

- [ ] Per-stage medians, percentiles and maxima
- [ ] Deadline misses counted
- [ ] Procedure needs only the Docker image, a clone and the COMFI subset
- [ ] Dropped frames in COMFI recordings reported from timestamps where available
