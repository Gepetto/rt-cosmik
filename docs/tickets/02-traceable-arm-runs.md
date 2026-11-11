# 02: Traceable arm runs

**What to build:** Every arm run records commit, settings, fold, landmark map identity, hardware and date in its run information, and refuses to start when pipeline code has uncommitted changes. The arm evaluation seam gets a synthetic-trial test harness and a replay-versus-direct regression test.

**Blocked by:** None (can start immediately)

**Status:** implemented, awaiting the author's regression run (GPU) and commit

- [x] Run information contains commit, settings, fold, map identity, hardware, date
- [x] A run with uncommitted pipeline changes is refused
- [x] Synthetic-trial harness drives the arm evaluation seam end to end
- [ ] The regression test reproduces the submitted numbers for one participant, before and after the merge of main (ticket 03)
- [ ] Replay matches a direct run within the documented sub-millimetre median marker difference

## Open after review

- Fold is recorded as `None` until ticket 05 defines folds; ticket 05 must also make the landmark-map identity the fitted per-fold map, not only the queried vertex indices.
- The synthetic harness has no perturbation hooks yet (lag, noise, masking, outliers); tickets 04 and 10 add them with their tests.
- Generated solvers under `ocp/` are gitignored, so the dirty-tree check does not cover them; their fingerprint is checked separately by the solver loader.
- `run_pipeline.py` records provenance but does not refuse a dirty tree, so developers can still run the toolbox.
- Pre-existing failure, unrelated: `test_artifact_root_is_absolute_and_overridable` expects no `_parity` suffix; settles with the merge (ticket 03).
