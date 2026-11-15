# 19: Sustained-replay timing

**What to build:** A sustained-replay mode recording per-stage timestamps (decode, detection, NLF, fusion, filter, IK, output) and missed 40 Hz deadlines, runnable from main's Docker image plus a documented COMFI subset. It is developed and first measured on the workstation GPU. The RTX 3050 Ti laptop and the RTX 4070 desktop are measured by the author at the very end, from a clean install.

**Blocked by:** 22 (done). Not blocked by 18: the timing mode does not need the simplification.

**Status:** done on the workstation (2026-11-15): `scripts/python/paper/sustained_timing.py`, procedure in `docs/research/timing.md`, results in `docs/research/timing-workstation.md`. The laptop and the RTX 4070 desktop remain, by the author, at the very end.

- [x] Per-stage medians, percentiles and maxima
- [x] Deadline misses counted
- [x] Procedure needs only the Docker image, a clone and the COMFI subset
- [x] Dropped frames in COMFI recordings reported from timestamps where available
- [x] Solve time of the MHE and of the per-frame IK measured in the same idle-machine session (the submitted "1.7 ms of additional computation")
- [x] The timing table states that rates are processing rates on recorded video, capped in practice by the 40 Hz camera rate

**End-to-end latency (asked by two reviewers).** A replay cannot measure the acquisition delay, the inter-camera synchronization error of the software synchronization, or the true camera-to-output latency. Two steps, in this order:

- [x] Option 1: report latency as filter lag plus processing time plus at most one frame of waiting, with its percentiles from the sustained replay, and state that acquisition and synchronization delays are not measured. Check first whether the COMFI recordings store per-camera timestamps; if they do, report inter-camera synchronization error and dropped frames from them.
- [ ] Option 2 (not needed: the COMFI recordings store per-camera timestamps), only if the timestamps are missing and only with the author's agreement: one short live timing run with the cameras, no participant and no new dataset, recording timestamps only. It is an exception to the rule of no new live runs.
