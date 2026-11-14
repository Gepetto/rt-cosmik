# 09: 1€ filter arm and headline rule

**What to build:** A causal 1€ filter on the fused landmarks, tuned per fold to minimize causal error with jitter at or below the 5 Hz Butterworth's. The headline-arm rule (MHE without filter unless the 1€ arm has lower causal error) is recorded before out-of-fold results are computed.

**Blocked by:** 04, 07

**Status:** done; rule recorded below before any 1-euro run was scored

**Rule, fixed before the 1-euro results exist.** Grid: cutoff at rest 0.5, 1, 2, 3, 5 Hz by beta 0.5, 2, 5, 15 Hz per m/s, each coordinate of each fused landmark filtered on its own, derivative cutoff 1 Hz. For each fold, on its fifteen fitting participants, the chosen point is the one with the lowest causal error among those whose jitter does not exceed that of the 5 Hz order-4 Butterworth on the same participants. The cross-validated 1-euro arm takes each participant from its own fold's choice. The headline arm is the MHE without filter unless the cross-validated 1-euro arm has a lower mean causal error over the 18 participants; if a fold has no admissible point, the headline is the MHE without filter. The no-filter, Butterworth and horizon results were already known when this was written (8.45 deg causal without filter, 8.70 deg with the 5 Hz Butterworth); the 1-euro results were not.

- [x] Filter is causal and fed one frame at a time
- [x] Parameters fitted per fold on the other fifteen participants
- [x] Headline rule committed before the results it selects

**Outcome (2026-11-14).** Every fold chose 2 Hz at rest with beta 15. Cross-validated 1-euro arm: 8.462 deg causal, jitter 0.153 deg, lag 0.4 frame. MHE without filter: 8.454 deg, jitter 0.279 deg, lag -0.4 frame. 5 Hz Butterworth: 8.700 deg, jitter 0.156 deg, lag 2.9 frames. By the rule the headline is the MHE without filter, by 0.007 deg; the 1-euro arm has the Butterworth's jitter at the no-filter accuracy.

**Decision of the author (2026-11-14).** The headline results use the tuned 1-euro filter. The paper states that no filter, the 5 Hz Butterworth and the tuned 1-euro filter were compared, and that the 1-euro filter gives less jitter than the Butterworth without its lag and the loss of causal accuracy the lag causes. For a like-for-like comparison, a Butterworth is tuned by the same per-fold rule over orders 1, 2 and 4.
