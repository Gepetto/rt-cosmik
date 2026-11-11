# 05: Folds and guarded landmark maps

**What to build:** A fixed six-fold assignment of the 18 participants (by ID only). NLF and FastSAM-3D landmark maps are fitted per fold on the other fifteen participants with the existing procedure; hand-picked landmarks stay fixed. The arm runner refuses a map whose fitting set contains the evaluated participant.

**Blocked by:** 02, 03

**Status:** code done; maps to be fitted by the author (commands in `scripts/python/paper/README.md`)

- [x] Fold assignment versioned and used by every fitting step
- [ ] Six NLF maps and six FastSAM-3D maps produced (FastSAM-3D from camera-2 meshes; 4687 left out of its fits, as the learner drops participants missing a task: RobotWelding is one frame longer than its reference)
- [x] Scoring a participant with its own fold's map is refused (test at the arm seam)
- [x] No participant name appears in any new file
