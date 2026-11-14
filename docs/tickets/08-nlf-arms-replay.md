# 08: Out-of-fold NLF arms by replay

**What to build:** Every NLF arm rebuilt from the recorded views: camera configurations, NLF-2D triangulation, fusion variants, horizon, filter settings, MHE weight sensitivity, and per-frame IK with matched inputs and filtering. These produce the main tables.

**Blocked by:** 04, 07

**Status:** done (runs in /root/workspace/oof/runs26, tables in /root/workspace/oof/t26/results; findings section 0)

- [x] All arms scored out-of-fold with causal and aligned error
- [x] Per-frame IK uses exactly the headline arm's inputs and filter
- [x] MHE weight sensitivity reported
