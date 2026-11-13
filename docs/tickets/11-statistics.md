# 11: Statistics

**What to build:** Holm correction within each family of tests, FastSAM-3D included, and two one-sided tests against a ±1° margin for equivalence claims. Built on the re-scored existing results, re-run on the final ones.

**Blocked by:** 04

**Status:** families `ik_filter` and `fusion` run on the out-of-fold studies; family `arms` waits for the out-of-fold arms (tickets 06, 08)

- [x] Every reported p-value belongs to a declared family with Holm correction
- [x] Equivalence claims use TOST at ±1°
