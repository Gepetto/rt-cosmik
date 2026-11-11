# 05: Folds and guarded landmark maps

**What to build:** A fixed six-fold assignment of the 18 participants (by ID only). NLF and FastSAM-3D landmark maps are fitted per fold on the other fifteen participants with the existing procedure; hand-picked landmarks stay fixed. The arm runner refuses a map whose fitting set contains the evaluated participant.

**Blocked by:** 02, 03

**Status:** ready-for-agent

- [ ] Fold assignment versioned and used by every fitting step
- [ ] Six NLF maps and six FastSAM-3D maps produced
- [ ] Scoring a participant with its own fold's map is refused (test at the arm seam)
- [ ] No participant name appears in any new file
