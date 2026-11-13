# 24: InstantHMR and CPU feasibility

**What to build:** A one-day test, on existing COMFI videos only, of whether InstantHMR (medium, one-crop fit) can replace NLF-S as a faster front end and make a CPU configuration real time. It runs through the same fusion, filter, IK and scoring as the NLF arm on one fold, and its speed is measured on the workstation GPU and CPU next to NLF-S. The test ends with a decision: add it as an arm (ticket 25) or drop it.

**Blocked by:** 21 (comes after every ticket planned for the resubmission)

**Status:** ready-for-agent

- [ ] InstantHMR medium runs on COMFI crops with the calibrated focal length and returns camera-frame metric landmarks
- [ ] A landmark map on its mesh is fitted on fifteen participants and used on the three of one fold (18 trials)
- [ ] Causal and aligned joint error, marker error and jitter on those 18 trials, next to the NLF-S arm on the same trials
- [ ] Per-stage timing on the workstation GPU (TensorRT), one to four views
- [ ] CPU timing with ONNX Runtime and OpenVINO, for InstantHMR and for NLF-S, with the thread count stated
- [ ] The licence of the weights used is stated (the accurate variants are non-commercial)
- [ ] Decision recorded: accuracy within a few tenths of a degree of NLF-S and about 30 fps on CPU, or dropped
