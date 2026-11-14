# 10: Fusion validation

**What to build:** Per-landmark weighted-median fusion added; reliability of NLF's σ against actual error per landmark and view; views randomly masked and synthetic outliers injected into one view; robot tasks reported separately.

**Blocked by:** 04, 07

**Status:** done (branch `ticket-10`; results in `docs/research/fusion-validation.md`)

- [x] Weighted median resists an outlier view on a synthetic trial
- [x] Inverse-variance beats the mean when σ is truthful and not when it is uninformative (synthetic)
- [x] Reliability analysis per landmark and view
- [x] Handling of missing landmarks, extreme weights and outliers documented
