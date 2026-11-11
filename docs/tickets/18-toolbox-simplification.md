# 18: Toolbox simplification

**What to build:** Main already removed the extra front ends, pinned dependencies, documented the code and added Docker (merged in ticket 03). What remains: the paper code (RTMPose+LSTM, FastSAM-3D, studies) kept in a clearly separated paper area, human-pose calibration part of the install, and one documented calibrate → run → evaluate path; arm choices in settings, paths only on the command line.

**Blocked by:** 03, 14

**Status:** ready-for-agent

- [ ] Clean install from the documented steps runs the pipeline
- [ ] Replay-versus-direct regression test still passes
- [ ] Paper studies still run from their area
