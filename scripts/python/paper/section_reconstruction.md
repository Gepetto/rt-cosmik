# Draft section: 3D reconstruction from multiple views

Replaces §III.B ("Stereo fusion and temporal denoising of 3D keypoints") of the
current draft. Written to describe what `rtcosmik/triangulation/triangulation.py`
actually does, for both modalities, and why they behave differently.

Symbols: `C` cameras, `J` landmarks. Camera `c` has intrinsics `K_c`,
distortion `d_c`, and extrinsics `[R_c | t_c]` mapping a point `p` in the
reference frame (camera 0) into camera `c`: `p_c = R_c p + t_c`.

---

## B. Recovering 3D from multiple views

The two input modalities reach 3D by different routes, and the difference is not
incidental — it determines how the reconstruction degrades as cameras are removed
or badly placed. We describe both, then quantify the geometry that separates them.

### 1) 2D multi-view: weighted linear triangulation

The 2D modality detects `J` image landmarks per view. Each is undistorted to
normalised coordinates `x_{c,j}` with `K_c, d_c`. A landmark's image position
constrains the 3D point to a ray, contributing two linear equations in the
homogeneous point `X_j`:

    ( y_{c,j} P_c^3 - P_c^2 ) X_j = 0
    ( P_c^1 - x_{c,j} P_c^3 ) X_j = 0

where `P_c^r` is row `r` of `[R_c | t_c]`. Stacking both rows for every view
gives `A_j X_j = 0`, solved as the right null vector of `A_j` by SVD.

Each view's two rows are scaled by a per-landmark weight

    w_{c,j} = ( 1 / sigma_{c,j} )^2,   sigma_{c,j} = 1 / max(s_{c,j}, s_min)

with `s_{c,j}` the detector's confidence, normalised so the best view of each
landmark has unit weight. This is inverse-variance weighting on the algebraic
residual: a view that cannot see a landmark is suppressed *for that landmark
only* while still contributing everywhere else, and a weight of zero is exactly
equivalent to omitting that view from the system.

**What this does and does not use.** Triangulation uses only the direction of
each ray. Depth along the ray is not measured by any single view; it is recovered
solely from where the rays intersect. The accuracy of that intersection is set by
the angles between rays, not by the detector.

### 2) 3D multi-view: fusion of per-view metric poses

The 3D modality regresses a *metric* 3D pose per view: view `c` returns
`p_{c,j}` in its own camera frame, together with a per-landmark uncertainty
`sigma_{c,j}`. Each pose is mapped into the reference frame,

    q_{c,j} = R_c^T ( p_{c,j} - t_c )

and the views are combined per landmark by inverse-variance weighting

    X_j = ( sum_c w_{c,j} q_{c,j} ) / ( sum_c w_{c,j} ),   w_{c,j} = sigma_{c,j}^-2

**There is no triangulation step.** No ray is intersected and no epipolar
constraint is imposed. Each view already carries a complete estimate, including
range, obtained from the network's learned body model rather than from parallax;
fusion combines whole estimates rather than reconstructing one from partial ones.

Two properties follow, and both are visible in the results. First, each view's
pose is anatomically coherent — segment lengths come from a body model — so
fusion preserves that coherence, whereas triangulating landmarks independently
imposes nothing between them. Second, many tracked landmarks are *internal*
points (hip joint centres, spine) that no camera observes directly; a regressed
body model places them consistently, while triangulation must intersect rays
toward points that were never seen.

### 3) Why the two behave differently as the rig changes

The distinction matters because the two schemes have opposite geometric
preferences.

**Triangulation needs angular diversity.** With isotropic image noise, the
information a set of views carries about `X` is

    F = sum_c ( I - u_c u_c^T ) / r_c^2

with `u_c` the unit ray from camera `c` to the point and `r_c` its range. A view
constrains the two directions perpendicular to its ray and contributes nothing
along it. Two cameras facing one another see the subject along nearly the same
line, so neither constrains position along it.

**Monocular fusion needs directional diversity of a different kind.** A
monocular range error is a signed displacement along the viewing axis. Averaging
views whose axes point in opposite directions cancels that bias; averaging views
that look from the same side does not, because their errors are parallel.

For the rig used here — two stereo pairs facing each other across 5.3 m, the
subject between them — rays are 17 deg apart within a pair and 148 to 165 deg
apart across pairs. The eigenvalues of `F` are `[0.19, 3.84, 3.97]`: the weakest
direction carries roughly twenty times less information than the other two, and
lies within 6 to 9 deg of the reference camera's optical axis. The same
arrangement that makes triangulation ill-conditioned along that axis is the one
that best cancels monocular range bias.

This is measurable in both directions. Splitting marker error into a component
along the reference camera's optical axis and a component perpendicular to it,
the 2D modality is anisotropic at every camera count (1.70 to 1.85), while the
3D modality reaches 1.02 with four cameras — depth resolved as well as any other
direction. Conversely, the 3D modality's depth error falls from 104.2 mm with one
camera to 93.3 mm with a same-side pair (worse than the sqrt(2) that independent
errors would give) and to 35.6 mm once the opposing pair is added (better than
sqrt(4)), which is the signature of systematic cancellation rather than
averaging.

### 4) Validation of the triangulation implementation

To confirm that the 2D modality is limited by its input rather than by our
implementation, we projected reference marker trajectories through the measured
calibration, perturbed the projections with Gaussian pixel noise, and
triangulated them back. With no noise the points are recovered to 0.000 mm. With
the noise level the 2D detector actually delivers on this data (about 15 px), the
recovered depth error is 73.1 mm against a measured 73.5 mm, and the anisotropy
2.04 against a measured 2.10. We further compared four estimators — the weighted
linear solve above, an unweighted solve, iterative refinement by projective
depth, a Huber-robust variant, and inverse-covariance fusion of the two stereo
pairs. Depth error varied between 70.8 and 75.5 mm and whole-body joint RMSE
between 12.47 and 12.56 deg. The 2D modality therefore operates at the limit its
geometry and its detector allow.

### 5) Temporal filtering

Reconstructed landmarks pass through a fourth-order zero-latency Butterworth
low-pass at 5 Hz (sampling 40 Hz) before inverse kinematics. Both modalities use
the same filter with the same parameters.

---

## Notes for whoever edits this

- The reference-frame mapping `q = R^T (p - t)` is the inverse of the convention
  used for the projection matrices; keep them consistent if the notation changes.
- The claim "no triangulation step" is worth keeping explicit. It is the point a
  reader most reliably misreads, since the 3D modality also produces 2D landmarks
  and it is natural to assume those are triangulated.
- The caveat in §3 cuts both ways and should stay: on a rig with pairs at 90 deg
  the predicted anisotropy falls from 6.66 to 1.41, so triangulation would improve
  substantially while the 3D modality's bias cancellation would weaken. The gap
  reported here is partly a property of this camera arrangement.
