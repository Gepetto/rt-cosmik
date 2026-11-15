# 21: Rebuild handoff, tables and figures

**What to build:** Every number, table and figure of the revised paper regenerated from out-of-fold, traceable runs, and the response's blue numbers updated.

**Blocked by:** 06, 08, 09, 10, 11, 12, 13, 15, 16, 17, 19

**Status:** done (2026-11-15): `scripts/python/paper/paper_tables.py` writes `docs/research/paper-tables.md` (tables, with the commit and folds of the runs behind each arm); `trace_grid.py` the trace figure. Open: Fig. 5 (representative postures) still shows the submitted estimates; laptop and RTX 4070 timings.

- [x] Each paper number traces to a run with commit and fold
- [x] Response to reviewers updated with final numbers
