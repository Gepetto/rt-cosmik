import numpy as np
import matplotlib.pyplot as plt
import math


import pandas as pd



def fuse_stereo_keypoints_vel(
    P_L, P_R,                  # (K,3) triangulated from (0,2) and (4,6)
    score_cam0, score_cam2,    # (K,)  confidences for cameras 0,2
    score_cam4, score_cam6,    # (K,)  confidences for cameras 4,6
    T_c0, T_c2, T_c4, T_c6,    # (4,4) camera world poses
    sigma=None,                # length-scale for distance weighting (meters)
    gamma=1.0,                 # exponent for confidence weighting
    eps=1e-8,
    # ---- NEW: optional velocity-based downweighting (past-only) ----
    prev_P_L=None, prev_P_R=None,  # (K,3) triangulations from previous frame
    dt=None,                       # seconds between frames
    vel_sigma=2.0,                 # m/s; scalar or (K,) — typical max speed before downweight
    vel_power=1.0,                 # how strongly velocity affects the weight
    v_cap=None                     # optional cap on |v| in m/s to avoid extreme spikes
):
    """
    Returns:
      fused: (K,3) fused 3D points
      score: (K,) normalized fusion strength in [0,1]

    Base weights (unchanged):
      w_L ∝ (conf_L^gamma) * exp( -||P_L - cL||^2 / (2*sigma^2) ), idem for right.

    Added:
      Per-marker velocity weights for left/right:
        w_vel_L[i] = exp(-0.5 * (||v_L[i]|| / vel_sigma[i])^2) ** vel_power
        w_vel_R[i] = exp(-0.5 * (||v_R[i]|| / vel_sigma[i])^2) ** vel_power
      with v_* computed from previous triangulation only (past data).
    """

    P_L = np.asarray(P_L, float)
    P_R = np.asarray(P_R, float)
    K = P_L.shape[0]

    # Camera centers (world)
    p0 = np.asarray(T_c0[:3, 3], float)
    p2 = np.asarray(T_c2[:3, 3], float)
    p4 = np.asarray(T_c4[:3, 3], float)
    p6 = np.asarray(T_c6[:3, 3], float)
    cL = 0.5 * (p0 + p2)
    cR = 0.5 * (p4 + p6)

    # Pair confidences — use mean of the two cams, clipped to [0, 1+]
    score_cam0 = np.asarray(score_cam0, float)
    score_cam2 = np.asarray(score_cam2, float)
    score_cam4 = np.asarray(score_cam4, float)
    score_cam6 = np.asarray(score_cam6, float)
    conf_L = np.clip(0.5 * (score_cam0 + score_cam2), 0.0, None)  # (K,)
    conf_R = np.clip(0.5 * (score_cam4 + score_cam6), 0.0, None)  # (K,)

    # Distances of triangulations to their pair centers
    dL = np.linalg.norm(P_L - cL[None, :], axis=1)  # (K,)
    dR = np.linalg.norm(P_R - cR[None, :], axis=1)  # (K,)

    # Default sigma: half the distance between pair centers, with a floor
    if sigma is None:
        sigma = max(np.linalg.norm(cR - cL)/2, 0.5)  # >= 50 cm

    # Spatial kernels
    spat_L = np.exp(-0.5 * (dL / sigma) ** 2)
    spat_R = np.exp(-0.5 * (dR / sigma) ** 2)

    # Final weights (base)
    wL = (conf_L ** gamma) * spat_L
    wR = (conf_R ** gamma) * spat_R

    # ---- NEW: velocity-based per-marker weights (past-only, NaN-safe) ----
    if (prev_P_L is not None) and (prev_P_R is not None) and (dt is not None) and (dt > 0):
        prev_P_L = np.asarray(prev_P_L, float)
        prev_P_R = np.asarray(prev_P_R, float)

        okL = np.isfinite(P_L).all(axis=1) & np.isfinite(prev_P_L).all(axis=1)
        okR = np.isfinite(P_R).all(axis=1) & np.isfinite(prev_P_R).all(axis=1)

        vL = np.zeros_like(P_L)
        vR = np.zeros_like(P_R)
        vL[okL] = (P_L[okL] - prev_P_L[okL]) / dt
        vR[okR] = (P_R[okR] - prev_P_R[okR]) / dt

        speedL = np.linalg.norm(vL, axis=1)
        speedR = np.linalg.norm(vR, axis=1)

        if v_cap is not None:
            speedL = np.minimum(speedL, float(v_cap))
            speedR = np.minimum(speedR, float(v_cap))

        # Allow scalar or per-marker sigma
        vel_sigma_arr = np.asarray(vel_sigma, float)
        if vel_sigma_arr.ndim == 0:
            vel_sigma_arr = np.full(K, max(vel_sigma_arr, 1e-6), float)
        else:
            vel_sigma_arr = np.maximum(vel_sigma_arr, 1e-6)

        w_vel_L = np.ones(K, float)
        w_vel_R = np.ones(K, float)
        w_vel_L[okL] = np.exp(-0.5 * (speedL[okL] / vel_sigma_arr[okL])**2) ** float(vel_power)
        w_vel_R[okR] = np.exp(-0.5 * (speedR[okR] / vel_sigma_arr[okR])**2) ** float(vel_power)

        # Multiply into base weights
        wL *= w_vel_L
        wR *= w_vel_R
    # ---------------------------------------------------------------------

    # Handle invalid triangulations
    badL = ~np.isfinite(P_L).all(axis=1)
    badR = ~np.isfinite(P_R).all(axis=1)
    wL[badL] = 0.0
    wR[badR] = 0.0

    # Weighted fusion
    denom = wL + wR  # (K,)
    fused = np.empty_like(P_L)
    fused[:] = np.nan
    nz = denom > eps
    fused[nz] = (wL[nz, None] * P_L[nz] + wR[nz, None] * P_R[nz]) / denom[nz, None]

    # Fallbacks (only one side has non-negligible weight)
    onlyL = (wL > eps) & ~(wR > eps)
    onlyR = (wR > eps) & ~(wL > eps)
    fused[onlyL] = P_L[onlyL]
    fused[onlyR] = P_R[onlyR]

    # Normalized fusion score in [0,1]
    max_denom = np.max(denom[nz]) if np.any(nz) else 1.0
    score = np.zeros(K, dtype=float)
    if max_denom > eps:
        score[nz] = denom[nz] / (max_denom + eps)

    return fused, score

 

def fuse_stereo_keypoints(
    P_L, P_R,              # (K,3) triangulated from (0,2) and (4,6)
    score_cam0, score_cam2,# (K,)  confidences for cameras 0,2
    score_cam4, score_cam6,# (K,)  confidences for cameras 4,6
    T_c0, T_c2, T_c4, T_c6,# (4,4) camera world poses
    sigma=None,            # length-scale for distance weighting (meters)
    gamma=1.0,             # exponent for confidence weighting
    eps=1e-8
):
    """
    Returns:
      fused: (K,3) fused 3D points
      score: (K,) normalized fusion strength in [0,1]
             (higher means better: high confidences and good spatial consistency)

    Weights:
      w_L ∝ (conf_L^gamma) * exp( -||P_L - cL||^2 / (2*sigma^2) ), idem for right.
      cL,cR are centers of camera pairs (0,2) and (4,6).
    """

    P_L = np.asarray(P_L, float)
    P_R = np.asarray(P_R, float)
    K = P_L.shape[0]

    # Camera centers (world)
    p0 = np.asarray(T_c0[:3, 3], float)
    p2 = np.asarray(T_c2[:3, 3], float)
    p4 = np.asarray(T_c4[:3, 3], float)
    p6 = np.asarray(T_c6[:3, 3], float)
    cL = 0.5 * (p0 + p2)
    cR = 0.5 * (p4 + p6)

    # Pair confidences — use mean of the two cams, clipped to [0, 1+]
    score_cam0 = np.asarray(score_cam0, float)
    score_cam2 = np.asarray(score_cam2, float)
    score_cam4 = np.asarray(score_cam4, float)
    score_cam6 = np.asarray(score_cam6, float)
    conf_L = np.clip(0.5 * (score_cam0 + score_cam2), 0.0, None)  # (K,)
    conf_R = np.clip(0.5 * (score_cam4 + score_cam6), 0.0, None)  # (K,)

    # Distances of triangulations to their pair centers
    dL = np.linalg.norm(P_L - cL[None, :], axis=1)  # (K,)
    dR = np.linalg.norm(P_R - cR[None, :], axis=1)  # (K,)

    # Default sigma: half the distance between pair centers, with a floor
    if sigma is None:
        sigma = max(np.linalg.norm(1*cR - cL)/2, 0.5)  # >= 50 cm

    # Spatial kernels
    spat_L = np.exp(-0.5 * (dL / sigma) ** 2)
    spat_R = np.exp(-0.5 * (dR / sigma) ** 2)

    # Final weights
    wL = (conf_L ** gamma) * spat_L
    wR = (conf_R ** gamma) * spat_R

    # Handle invalid triangulations
    badL = ~np.isfinite(P_L).all(axis=1)
    badR = ~np.isfinite(P_R).all(axis=1)
    wL[badL] = 0.0
    wR[badR] = 0.0

    # Weighted fusion
    denom = wL + wR  # (K,)
    fused = np.empty_like(P_L)
    fused[:] = np.nan
    nz = denom > eps
    fused[nz] = (wL[nz, None] * P_L[nz] + wR[nz, None] * P_R[nz]) / denom[nz, None]

    # Fallbacks (only one side has non-negligible weight)
    onlyL = (wL > eps) & ~(wR > eps)
    onlyR = (wR > eps) & ~(wL > eps)
    fused[onlyL] = P_L[onlyL]
    fused[onlyR] = P_R[onlyR]

    # Normalized fusion score in [0,1]
    max_denom = np.max(denom[nz]) if np.any(nz) else 1.0
    score = np.zeros(K, dtype=float)
    if max_denom > eps:
        score[nz] = denom[nz] / (max_denom + eps)
    # where both weights are ~0 (no valid triangulation), score stays 0

    return fused, score




def init_kf_state(K, dt=1/30, q=5e-3, r=5e-4, gate_sigma=4.0,
                  bones=None, ref_len=None, bone_alpha=0.5, n_proj_iters=2,
                  z0=None):
    """
    Initialize state for per-joint constant-velocity KF (+ soft bone projection).
    K          : number of joints
    bones      : list of (i,j) index pairs; None to disable projection
    ref_len    : array of target lengths (len(bones),); None -> projection disabled
    z0         : optional (K,3) initial positions
    """
    I3 = np.eye(3); dt = float(dt)
    F  = np.block([[I3, dt*I3],
                   [np.zeros((3,3)), I3]])                 # 6x6
    H  = np.block([I3, np.zeros((3,3))])                  # 3x6
    Q  = q * np.block([[(dt**3/3)*I3, (dt**2/2)*I3],
                       [(dt**2/2)*I3,  dt*I3]])
    R  = r * I3

    X = np.zeros((K, 6), dtype=float)   # [pos(3), vel(3)]
    P = np.tile(np.eye(6), (K,1,1)).astype(float)
    if z0 is not None:
        z0 = np.asarray(z0, float)
        m = ~np.isnan(z0).any(axis=1)
        X[m, :3] = z0[m]

    state = dict(
        K=K, F=F, H=H, Q=Q, R=R, gate2=float(gate_sigma**2),
        X=X, P=P, bone_alpha=float(bone_alpha), n_proj_iters=int(n_proj_iters)
    )
 
    state["ref_len"] = np.asarray(ref_len, float)
 
    return state


def kf_keypoints_step(z, state, HPE2_MOCAP):
    """
    One update step. Input z: (K,3) noisy positions (NaNs allowed).
    Returns filtered positions (K,3) and updates `state` in-place.
    """
    z = np.asarray(z, float)
    K = state["K"]; F=state["F"]; H=state["H"]; Q=state["Q"]; R=state["R"]; gate2=state["gate2"]
    X=state["X"]; P=state["P"]
    
    Q_global = state["Q"]
    Qj = state.get("Qj", None)

    # ------------------------------------------------------------------------------------------

    # predict
    for j in range(K):
        Q_use = Qj[j] if Qj is not None else Q_global
        X[j] = F @ X[j]
        P[j] = F @ P[j] @ F.T + Q_use

    # update (skip NaNs; per-joint R if provided)
    I6 = np.eye(6)
    updated = np.zeros(K, dtype=bool)

    for j in range(K):
        meas = z[j]
        if np.any(np.isnan(meas)):
            continue

        Rj = state.get("Rj", R)[j]
        y  = meas - (H @ X[j])                # innovation
        S  = H @ P[j] @ H.T + Rj              # innovation cov

        HP = H @ P[j]
        try:
            S_inv_HP = np.linalg.solve(S, HP)
        except np.linalg.LinAlgError:
            S_inv_HP = np.linalg.lstsq(S, HP, rcond=None)[0]
        Kk = S_inv_HP.T

        X[j] = X[j] + Kk @ y

        KH  = Kk @ H
        Pj  = (I6 - KH) @ P[j] @ (I6 - KH).T + Kk @ Rj @ Kk.T  # use Rj here
        P[j] = 0.5 * (Pj + Pj.T)
        updated[j] = True

    # positions after KF
    Pnow = X[:, :3].copy()

    # -------------------- Soft projection to fixed arm lengths (mocap indices) --------------------
    ref_len = state["ref_len"]
    
    I_idx = np.array([
        HPE2_MOCAP["LShoulder"][1],  # L upper arm: shoulder -> elbow
        HPE2_MOCAP["LElbow"][1],     # L lower arm: elbow -> wrist
        HPE2_MOCAP["RShoulder"][1],  # R upper arm: shoulder -> elbow
        HPE2_MOCAP["RElbow"][1],     # R lower arm: elbow -> wrist
    ], dtype=int)

    J_idx = np.array([
        HPE2_MOCAP["LElbow"][1],
        HPE2_MOCAP["LWrist"][1],
        HPE2_MOCAP["RElbow"][1],
        HPE2_MOCAP["RWrist"][1],
    ], dtype=int)
    
    
    
    
    if (I_idx.size > 0) and (ref_len is not None) and state["n_proj_iters"] > 0 and state["bone_alpha"] > 0:
        B = len(I_idx)
        for _ in range(state["n_proj_iters"]):
            vi = Pnow[I_idx]                      # (B,3)
            vj = Pnow[J_idx]                      # (B,3)
            v  = vj - vi
            d  = np.linalg.norm(v, axis=1)        # (B,)
            u  = np.zeros_like(v)
            nz = d > 1e-12
            u[nz] = v[nz] / d[nz,None]
            scale = state["bone_alpha"] * 0.5 * (ref_len - d)   # (B,)
            delta = (scale[:,None]) * u
            np.add.at(Pnow, I_idx, -delta)
            np.add.at(Pnow, J_idx, +delta)

        # keep state consistent
        X[:, :3] = Pnow
    # ---------------------------------------------------------------------------------------------

    return Pnow






# def kf_keypoints_step(z, state):
#     """
#     One update step. Input z: (K,3) noisy positions (NaNs allowed).
#     Returns filtered positions (K,3) and updates `state` in-place.
#     """
#     z = np.asarray(z, float)
#     K = state["K"]; F=state["F"]; H=state["H"]; Q=state["Q"]; R=state["R"]; gate2=state["gate2"]
#     X=state["X"]; P=state["P"]
    
#     Q_global = state["Q"]
#     Qj = state.get("Qj", None)
     
#     # predict
#     for j in range(K):
#         Q_use = Qj[j] if Qj is not None else Q_global
#         X[j] = F @ X[j]
#         P[j] = F @ P[j] @ F.T + Q_use

#     # update (skip NaNs, Mahalanobis gating; use solve instead of inverse)
   
#     I6 = np.eye(6)
#     updated = np.zeros(K, dtype=bool)  # track who actually updated

#     for j in range(K):
#         meas = z[j]
#         if np.any(np.isnan(meas)):
#             continue
        
#         Rj = state.get("Rj", R)[j]   # <— per-joint R if provided
       
#         y  = meas - (H @ X[j])  # innovation (3,)
#         S  = H @ P[j] @ H.T + Rj  # innovation cov (3,3)
            
#         # y = meas - (H @ X[j])              # innovation (3,)
#         # S = H @ P[j] @ H.T + R             # innovation cov (3,3)

#         # Mahalanobis gate
#         # try:
#         #     v = np.linalg.solve(S, y)
#         # except np.linalg.LinAlgError:
#         #     v = np.linalg.lstsq(S, y, rcond=None)[0]
#         # md2 = float(y @ v)
#         # if md2 > gate2:
#         #     # optional: debug drift for joint 9
#         #     if j == 9:
#         #         print(f"[KF] j=9 gated: md2={md2:.2f} > {gate2:.2f}")
#         #     continue

#         # Kalman gain: K = P H^T S^{-1}
#         HP = H @ P[j]                       # (3,6)
#         try:
#             S_inv_HP = np.linalg.solve(S, HP)
#         except np.linalg.LinAlgError:
#             S_inv_HP = np.linalg.lstsq(S, HP, rcond=None)[0]
#         Kk = S_inv_HP.T                     # (6,3)

#         # state update
#         X[j] = X[j] + Kk @ y

#         # Joseph form for P (keeps PSD & symmetry)
#         KH  = Kk @ H                        # (6,6)
#         Pj  = (I6 - KH) @ P[j] @ (I6 - KH).T + Kk @ R @ Kk.T
#         P[j] = 0.5 * (Pj + Pj.T)            # re-symmetrize
#         updated[j] = True

   

#     Pnow = X[:, :3].copy()

#     # soft bone-length projection (vectorized)
#     I_idx, J_idx, ref_len = state["I_idx"], state["J_idx"], state["ref_len"]
#     if (I_idx is not None) and (ref_len is not None) and state["n_proj_iters"] > 0 and state["bone_alpha"] > 0:
#         B = len(I_idx)
#         for _ in range(state["n_proj_iters"]):
#             vi = Pnow[I_idx]                 # (B,3)
#             vj = Pnow[J_idx]                 # (B,3)
#             v  = vj - vi                     # (B,3)
#             d  = np.linalg.norm(v, axis=1)   # (B,)
#             u  = np.zeros_like(v)
#             nz = d > 1e-12
#             u[nz] = v[nz] / d[nz,None]
#             scale = state["bone_alpha"] * 0.5 * (ref_len - d)   # (B,)
#             delta = (scale[:,None]) * u
#             np.add.at(Pnow, I_idx, -delta)
#             np.add.at(Pnow, J_idx, +delta)
#         # keep state consistent
#         X[:, :3] = Pnow
 
        
        
        
#     return Pnow


def to_utc(s):
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")




def plot_3D_keypoints_and_scores(IDX,scores_cam1,scores_cam2):
 

    names = list(IDX.keys())
    K = len(names)
    cols = 6
    rows = math.ceil(K / cols)

    vmin, vmax = 0.5, 2.0  # red <= 0.3, green >= 1.0
    cmap = plt.get_cmap('RdYlGn')  # low=red, high=green
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.0, rows*2.2), sharex=True)
    axes = axes.ravel()

    for i, name in enumerate(names):
        col = IDX[name]
        y =  scores_cam1[:, col]+scores_cam2[:, col]
        t = np.arange(len(y))

        ax = axes[i]
        # color-coded scatter by value
        ax.scatter(t, y, c=y, cmap=cmap, norm=norm, s=6)
        # optional thin line to show trend
        ax.plot(t, y, lw=0.6, color='0.4', alpha=0.5)

        ax.axhline(vmin, color='r', ls='--', lw=0.6, alpha=0.6)
        ax.axhline(vmax, color='g', ls='--', lw=0.6, alpha=0.6)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        

    # single colorbar for all subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9, pad=0.02)
    cbar.set_label('covariance / confidence')

    axes[0].set_xlabel("frame")
    plt.tight_layout()
    plt.show()
    
    
def score_to_rgba(s, lo=0.5, hi=2.0):
    """
    Map confidence score s to RGBA.
    s <= lo  -> red;  s >= hi -> green; linear in between.
    """
    if np.isnan(s):
        return [0.5, 0.5, 0.5, 0.2]  # faint gray for missing
    t = (s - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    r = 1.0 - t
    g = t
    b = 0.0
    a = 1.0
    return [r, g, b, a]


def kabsch_global(P_cam_seq, P_mocap_seq, weights=None):
    """
    P_cam_seq, P_mocap_seq: arrays (T, N, 3) alignés temporellement et par point.
    Calcule UN seul (R,t) qui aligne tout (cam -> mocap) en minimisant la somme des erreurs.
    """
    assert P_cam_seq.shape == P_mocap_seq.shape and P_cam_seq.shape[-1] == 3
    T, N, _ = P_cam_seq.shape
    X = P_cam_seq.reshape(T*N, 3)
    Y = P_mocap_seq.reshape(T*N, 3)

    if weights is not None:
        w = np.asarray(weights).reshape(T, N)
        w = w / (w.sum() + 1e-12)
        w = w.reshape(T*N, 1)
        Xc = (X * w).sum(axis=0)     # weighted means
        Yc = (Y * w).sum(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = (Y0 * w).T @ X0
    else:
        Xc = X.mean(axis=0)
        Yc = Y.mean(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = Y0.T @ X0

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # corrige réflexion
        U[:, -1] *= -1
        R = U @ Vt
    t = Yc - R @ Xc

    X_align = (R @ X.T).T + t
    #rms = np.sqrt(np.mean(np.sum((X_align - Y)**2, axis=1)))
    return R, t#, rms

def mpjpe(jcp, jcp_hpe):
    """
    Mean Per Joint Position Error (MPJPE)
    jcp, jcp_hpe: arrays of shape (N, J, 3)
    Returns: scalar (average 3D joint error)
    """
    # Euclidean distance per frame & joint
    errors = np.linalg.norm(jcp - jcp_hpe, axis=2)  # (N, J)
    return errors.mean()


 

def plot_ref_est_ft_concatenated(
    ref,          # (N, J, 3)  in REF/MOCAP order
    est,          # (N, J, 3)  already reordered to REF order
    ft,           # (N, J, 3)  already reordered to REF order
    mapping=None, # IGNORED for indexing; only used for optional names
    ft_in_ref_order=True,   # IGNORED
    title=None
):
    """
    Reordering is REMOVED. Assumes ref, est, ft are already aligned: shape (N, J, 3), same J and order.
    'mapping' is only used (optionally) to display joint names; it does not affect indexing.
    """

    N, J, _ = ref.shape
    # Optional names (from mapping keys) purely for labels; fallback to J0..J{J-1}
    if isinstance(mapping, dict) and len(mapping) > 0:
        joint_names = list(mapping.keys())
        
        if len(joint_names) != J:
            joint_names = [joint_names[j] if j < len(joint_names) else f"J{j}" for j in range(J)]
    else:
        joint_names = [f"J{j}" for j in range(J)]

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    offset = 0
    boundaries = []
    name_positions = []
    rmse_points_est = []
    rmse_points_ft  = []

    for j in range(J):
        ref_j = ref[:, j, :]  # (N,3)
        est_j = est[:, j, :]
        ft_j  = ft[:,  j, :]

        t = np.arange(N) + offset

        # Plot x,y,z
        for d, ax in enumerate(axes[:3]):
            ax.plot(t, ref_j[:, d], "k-", linewidth=1, label="ref" if (offset == 0 and d == 0) else None)
            ax.plot(t, est_j[:, d], "r-", linewidth=1, label="est" if (offset == 0 and d == 0) else None)
            ax.plot(t, ft_j[:,  d], "g-", linewidth=1, label="fine-tuned" if (offset == 0 and d == 0) else None)
            if d == 0:
                

                name_positions.append((offset + N/2, joint_names[j]))

        # Framewise residual norms
        res_est = np.linalg.norm(est_j - ref_j, axis=1)
        res_ft  = np.linalg.norm(ft_j  - ref_j, axis=1)

        # Per-joint RMSE (scalar over all coords & frames)
        rmse_est = np.sqrt(np.mean((est_j - ref_j)**2))
        rmse_ft  = np.sqrt(np.mean((ft_j  - ref_j)**2))

        axes[3].plot(t, res_est, color="r", linewidth=0.6)
        axes[3].plot(t, res_ft,  color="g", linewidth=0.6, alpha=0.35)

        center = offset + N/2
        rmse_points_est.append((center, rmse_est, joint_names[j]))
        rmse_points_ft.append((center, rmse_ft,  joint_names[j]))

        offset += N
        boundaries.append(offset)

    # Vertical boundaries
    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)

    # Joint name labels on top subplot
    top_ax = axes[0]
    y_top = top_ax.get_ylim()[1]
    for x, name in name_positions:
        top_ax.text(x, y_top, name, ha="center", va="top", fontsize=8, rotation=90)

    # Labels/legend
    for d, ax in enumerate(axes[:3]):
        ax.set_ylabel(["x","y","z"][d])
        if d == 0:
            ax.legend(loc="upper left", ncol=3, frameon=False)

    axes[3].set_ylabel("Residual / RMSE")
    axes[3].set_xlabel("Frames concatenated per marker")
    axes[3].legend(["est residual", "fine-tuned residual"], loc="upper left", frameon=False)

    plt.tight_layout()
    plt.show()

    
    
    
    
    
def estimate_joint_axis_accel_variances(mocap_xyz, dt, shrink=0.1, eps=1e-12):
    """
    mocap_xyz: array (N, K, 3) positions in meters.
    dt:        timestep in seconds.
    Returns:
      var_ax: (K,3) acceleration variances per joint & axis, from central diff.
              Uses shrinkage on the sample covariance for stability (diagonal kept).
    """
    N, K, _ = mocap_xyz.shape
    if N < 3:
        raise ValueError("Need at least 3 frames for central differences.")

    # central finite-difference accelerations (N-2, K, 3)
    acc = (mocap_xyz[2:] - 2*mocap_xyz[1:-1] + mocap_xyz[:-2]) / (dt**2)

    # estimate cov across time per joint, keep diagonal (axis-wise variance)
    # we add a tiny shrinkage to stabilize short sequences / jitter
    var_ax = np.var(acc, axis=0, ddof=1)     # (K,3), unbiased
    mean_var = np.mean(var_ax)
    var_ax = (1.0 - shrink) * var_ax + shrink * mean_var  # scalar shrinkage
    var_ax = np.maximum(var_ax, eps)
    return var_ax  # (K,3)
    

def cv_Q_axis(var_a, dt):
    """
    CV model (state [p, v]) driven by white acceleration with PSD = var_a.
    Returns the 2x2 block for a single axis.
    """
    dt2 = dt*dt
    return var_a * np.array([[dt2*dt/3.0, dt2/2.0],
                             [dt2/2.0,     dt      ]], dtype=float)


def build_per_joint_Q(var_ax, dt):
    """
    var_ax: (K,3) per-joint acceleration variances for x,y,z.
    Returns Qj: (K,6,6), each 6x6 in order [px,py,pz,vx,vy,vz].
    """
    K = var_ax.shape[0]
    Qj = np.zeros((K, 6, 6), dtype=float)
    for j in range(K):
        Qx = cv_Q_axis(var_ax[j, 0], dt)
        Qy = cv_Q_axis(var_ax[j, 1], dt)
        Qz = cv_Q_axis(var_ax[j, 2], dt)
        # place 2x2 blocks along (px,vx), (py,vy), (pz,vz)
        Qj[j][np.ix_([0,3],[0,3])] = Qx
        Qj[j][np.ix_([1,4],[1,4])] = Qy
        Qj[j][np.ix_([2,5],[2,5])] = Qz
        # (off-axis/process cross-terms omitted for simplicity)
    return Qj



import numpy as np

def remap_hpe_to_mocap(kp, mapping, J_out=None, fill=np.nan):
    """
    Remap HPE keypoints to MoCap order.

    kp:        (26,3) single frame  OR  (N,26,3) sequence
    mapping:   dict like HPE2_MOCAP where values are [hpe_idx, mocap_idx]
    J_out:     number of MoCap joints (default: inferred from mapping, e.g., 20)
    fill:      value for missing joints (default: NaN)

    returns:
      out: (20,3) or (N,20,3) arranged like jcp (MoCap order)
      missing: list of joint names without a mocap index in mapping
    """
    # collect valid pairs (mocap_idx, hpe_idx), skip entries without mocap index
    pairs = []
    missing = []
    for name, v in mapping.items():
        if len(v) >= 2 and v[1] is not None:
            hpe_idx, mocap_idx = v[0], v[1]
            pairs.append((mocap_idx, hpe_idx))
        else:
            missing.append(name)

    # sort by mocap index to ensure correct output order
    pairs.sort(key=lambda x: x[0])
    jcp_idx = [p[0] for p in pairs]
    hpe_idx = [p[1] for p in pairs]

    # output size
    if J_out is None:
        J_out = max(jcp_idx) + 1  # typically 20

    # allocate and fill
    if kp.ndim == 2:
        # (26,3) -> (20,3)
        out = np.full((J_out, 3), fill, dtype=kp.dtype)
        out[jcp_idx, :] = kp[hpe_idx, :]
    elif kp.ndim == 3:
        # (N,26,3) -> (N,20,3)
        N = kp.shape[0]
        out = np.full((N, J_out, 3), fill, dtype=kp.dtype)
        out[:, jcp_idx, :] = kp[:, hpe_idx, :]
    else:
        raise ValueError("kp must be (26,3) or (N,26,3)")

    return out, missing



def create_virtual_head_markers(HEAD, LEAR, REAR,
                                ear_up=0.02,  ear_ap=0.00,
                                head_down=0.02, head_ap=0.00):
    """
    Inputs (can be (3,) or (N,3)):
      HEAD, LEAR, REAR : marker positions in world coords.

    Offsets (meters):
      ear_up    : move LHD/RHD upward (superior)  along head vertical axis (+z)
      ear_ap    : move LHD/RHD forward (anterior) along head AP axis (+y)
      head_down : move FHD downward (inferior)    along head vertical axis (-z)
      head_ap   : move FHD forward (anterior)     along head AP axis (+y)

    Returns:
      FHD, LHD, RHD  (same shape as inputs)

    Head frame (right-handed):
      x: right->left  (REAR -> LEAR)
      z: superior     (component of HEAD - mid(ears) orthogonal to x)
      y: anterior     (y = z × x)
    """
    HEAD = np.asarray(HEAD, dtype=float)
    LEAR = np.asarray(LEAR, dtype=float)
    REAR = np.asarray(REAR, dtype=float)
    eps = 1e-12

    mid_ears = 0.5 * (LEAR + REAR)
    x = LEAR - REAR
    x /= (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

    up_guess = HEAD - mid_ears
    up_guess = up_guess - np.sum(up_guess * x, axis=-1, keepdims=True) * x
    z = up_guess / (np.linalg.norm(up_guess, axis=-1, keepdims=True) + eps)  # superior

    y = np.cross(z, x)  # anterior
    y /= (np.linalg.norm(y, axis=-1, keepdims=True) + eps)

    # Offsets (support scalars; if arrays of shape (N,), they will broadcast fine)
    FHD = HEAD - head_down * z + head_ap * y
    LHD = LEAR + ear_up   * z + ear_ap * y
    RHD = REAR + ear_up   * z + ear_ap * y
    return FHD, LHD, RHD


def make_virtual_shoulders(C7, RSHO, LSHO, up_offset=0.03, ap_offset=0.02):
    """
    Inputs (can be (3,) or (N,3)):
      C7, RSHO, LSHO : arrays of marker positions in world coords
    Offsets:
      up_offset : meters to move along the local superior (vertical) axis
      ap_offset : meters to move along the local anterior axis
    Returns:
      virtual_RSHO, virtual_LSHO  (same shape as inputs)
    Frame definition (right-handed):
      x: right->left  (from RSHO to LSHO)
      z: superior     (component of C7-midShoulders orthogonal to x)
      y: anterior     (y = z × x)
    """
    C7   = np.asarray(C7,   dtype=float)
    RSHO = np.asarray(RSHO, dtype=float)
    LSHO = np.asarray(LSHO, dtype=float)
    eps = 1e-12

    mid = 0.5 * (RSHO + LSHO)                                 # origin
    x   = LSHO - RSHO                                         # ML axis (R->L)
    x  /= (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

    up_guess = C7 - mid                                       # roughly superior
    up_guess = up_guess - np.sum(up_guess * x, axis=-1, keepdims=True) * x
    z   = up_guess / (np.linalg.norm(up_guess, axis=-1, keepdims=True) + eps)  # superior

    y   = np.cross(z, x)                                      # anterior (right-handed)
    y  /= (np.linalg.norm(y, axis=-1, keepdims=True) + eps)

    # Apply desired offsets in the local frame to each shoulder
    virtual_RSHO = RSHO + up_offset * z + ap_offset * y
    virtual_LSHO = LSHO + up_offset * z + ap_offset * y
    return virtual_RSHO, virtual_LSHO




def read_keypoints_from_csv(csv_path, scale=1.0):
    """
    Reads 3D keypoints from a CSV file and returns them in shape (N, K, 3).

    Parameters:
        csv_path (str): Path to the CSV file. Assumes columns are named like 'joint_x', 'joint_y', 'joint_z'.

    Returns:
        keypoints (np.ndarray): Array of shape (N, K, 3)
        joint_names (List[str]): List of joint names in the order found in the file
    """
    df = pd.read_csv(csv_path)

    # Extract base joint names (e.g., 'hip' from 'hip_x')
    joint_names = []
    for col in df.columns:
        if "_" in col:
            base, axis = col.rsplit("_", 1)
            if axis.lower() in ("x", "y", "z") and base not in joint_names:
                joint_names.append(base)

    N = len(df)
    K = len(joint_names)
    keypoints = np.empty((N, K, 3), dtype=float)

    for k, joint in enumerate(joint_names):
        keypoints[:, k, 0] = df[f"{joint}_x"].to_numpy(dtype=float) * scale
        keypoints[:, k, 1] = df[f"{joint}_y"].to_numpy(dtype=float) * scale
        keypoints[:, k, 2] = df[f"{joint}_z"].to_numpy(dtype=float) * scale

    return keypoints, joint_names

