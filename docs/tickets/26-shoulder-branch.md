# 26: Shoulder branch in the IK and in the scoring

**What to build:** The IK no longer leaves a shoulder pinned on a joint limit in the wrong angle branch, and joint error no longer depends on which of the two equivalent shoulder angle triples a run or the reference is in. The shoulder is flexion (Z), abduction (X), rotation (Y); `(z, x, y)` and `(z + 180, 180 - x, y + 180)` are the same rotation and both fit the limits. With one solver iteration per frame the IK stays in the branch it entered at 90 degrees of abduction, then a flexion or rotation limit holds it there and the arm stops following the markers (52 to 70 degrees of upper-arm orientation error). The reference does it too, in 7 arm-trials of SideOverhead.

**Blocked by:** None. It changes every joint angle result, so it comes before 08, 09, 10 and 21.

**Status:** code done; reference and arms to regenerate

- [x] When the newest node of the horizon has a shoulder angle within 3 degrees of a limit and the other branch fits the limits over the whole horizon with the same margin, the horizon moves to the other branch (states, and the sign of the abduction velocity and acceleration); a moved shoulder is left alone for 20 frames
- [x] Joint errors (causal, aligned, and the flip count) are measured with each shoulder of the run in whichever branch is nearer the reference's at that frame, differences wrapped
- [x] Reference runs can be read from another root (`RTCOSMIK_REFERENCE_DIR`), since the dataset is read-only; regenerated references go in the runs root next to the arms, where the figure and REBA scripts already look
- [x] On the eight trials tested outside the repo (rule compared with wider limits and with the unchanged solver, estimate and reference regenerated alike): error over the 29 DoF 11.9 to 10.1 degrees, shoulder orientation error 24 to 17 degrees, a trial without flips unchanged; wider flexion and rotation limits were rejected (a trial without flips went from 9.1 to 10.4 degrees)
- [ ] Reference regenerated for the 108 trials with this solver
- [ ] Headline, FastSAM-3D, fusion and study runs regenerated and the findings file updated
- [x] Freeing the flexion limit alone, as a rule-free alternative, was tested on the same eight trials and rejected: 10.9 degrees against 10.1 for the re-seed, the rotation limit still holds the shoulder
- [x] REBA reads shoulder flexion and abduction in the description with abduction within 90 degrees, and scores an arm above the horizontal as above shoulder level; on the regenerated reference and fusion runs no frame with abduction within 90 degrees changes score (0.2 to 0.4 % of frames change, all in SideOverhead)
