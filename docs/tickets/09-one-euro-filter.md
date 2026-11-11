# 09: 1€ filter arm and headline rule

**What to build:** A causal 1€ filter on the fused landmarks, tuned per fold to minimize causal error with jitter at or below the 5 Hz Butterworth's. The headline-arm rule (MHE without filter unless the 1€ arm has lower causal error) is recorded before out-of-fold results are computed.

**Blocked by:** 04, 07

**Status:** ready-for-agent

- [ ] Filter is causal and fed one frame at a time
- [ ] Parameters fitted per fold on the other fifteen participants
- [ ] Headline rule committed before the results it selects
