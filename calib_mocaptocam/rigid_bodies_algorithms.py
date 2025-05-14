import numpy as np

def challis(x, y):
    """
    Compute a uniform scale s, rotation R, and translation d that best fits
    the model:  y = s * R * x + d
   
    Args:
        x: (N,3) numpy array of 3D points
        y: (N,3) numpy array of 3D points

    Returns:
        R:  (3,3) rotation matrix with det(R) = +1
        d:  (3,)  translation vector
        s:  (float) uniform scale factor
        rms: (float) RMS of the residuals in 3D
    """
    # Number of points
    n = x.shape[0]
    if x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("x and y must both have shape (N, 3).")

    # 1. Compute centroids
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
   
    # 2. Center the data (shape becomes (3, N) after transpose)
    Xc = (x - mx).T
    Yc = (y - my).T
   
    # 3. Cross-dispersion matrix (optionally divide by n, as in Challis)
    #    The division by n does not affect the final R, but is convenient
    #    for computing s the same way as in the Challis paper.
    C = (Yc @ Xc.T) / n
   
    # 4. Compute SVD of C
    #    C = P * Sigma * Q.T
    P, Sigma, Q = np.linalg.svd(C, full_matrices=True)
   
    # 5. Enforce proper rotation with det(R) = +1
    #    We do this by adjusting the last column sign if needed.
    S_fix = np.eye(3)
    if np.linalg.det(P @ Q) < 0:
        S_fix[-1, -1] = -1.0
   
    R = P @ S_fix @ Q
   
    # 6. Compute uniform scale factor s
    #    Following the Challis approach, s = ( trace(R^T * C) ) / (average squared length in x).
    #    sum of squares of Xc = sum( Xc^2 ) = sum over i of ||x_i - mx||^2
    #    The factor 1/n in C and in sigmax2 cancels out neatly.
    sigmax2 = np.sum(Xc**2) / n  # average of squared coords in x
    # trace(R^T * C) = trace(C * R^T), same thing
    s = (1.0 / sigmax2) * np.trace(R.T @ C)
   
    # 7. Compute translation d
    d = my - s * R @ mx
   
    # 8. Calculate the RMS error
    #    Residual = s*R*x[i] + d - y[i]
    residuals = (s * R @ x.T).T + d - y
    sum_sq = np.sum(residuals**2)
    # Each point contributes 3 squared terms, so total dimension = 3*n
    rms = np.sqrt(sum_sq / (3 * n))
   
    return R, d, s, rms

# Unit test for the challis function
def test_challis():
    """Unit test for the challis function using known transformation parameters."""
    # Known transformation parameters
    R_known = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])  
    d_known = np.array([0.0, 5.0, 0.0])  # Translation vector
    s_known = 1.0  # Scale factor

    # Generate synthetic data
    x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    y = (s_known * (R_known @ x.T).T) + d_known  # Apply known transformation

    # Call the challis function
    R, d, s, rms = challis(x, y)
    print(R)
    print(d)
    print(s)

    # Assertions to check if the estimated parameters match the known ones
    assert np.allclose(R, R_known, atol=1e-3), f"Rotation matrix mismatch: {R} vs {R_known}"
    assert np.allclose(d, d_known, atol=1e-3), f"Translation vector mismatch: {d} vs {d_known}"
    assert np.isclose(s, s_known, atol=1e-3), f"Scale factor mismatch: {s} vs {s_known}"
    print(f"Test passed! RMS error: {rms:.6f}")

# Run the unit test
# test_challis()



import numpy as np
def soder(x, y):
    """
    Solve y = R x + d by SVD (Procrustes).
    x, y: shape (N,3)
    
    Returns:
        R  (3x3) : rotation matrix
        d  (3,)  : translation vector
        rms (float): RMS of residuals
    """
    # Mean (centroid) of each set
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    
    # Center the points
    Xc = (x - mx).T  # shape = (3, N)
    Yc = (y - my).T  # shape = (3, N)
    
    # Cross-dispersion matrix
    C = Yc @ Xc.T  # shape = (3, 3)
    
    # SVD
    P, _, Q = np.linalg.svd(C)
    
    # Enforce det(R) = +1
    S = np.eye(3)
    S[-1, -1] = np.linalg.det(P @ Q)
    R = P @ S @ Q
    
    # Translation vector
    d = my - R @ mx
    
    # Compute RMS error
    n = x.shape[0]
    # residuals: R*x[i] + d - y[i]
    residuals = (R @ x.T).T + d - y
    # Sum of squared residuals
    sum_sq = np.sum(residuals**2)
    # RMS is sqrt of average squared error in *all three* dimensions
    rms = np.sqrt(sum_sq / (3 * n))
    
    return R, d, rms

# Example usage (optional):
# x = np.array([[...], [...], ...])  # Replace with actual marker coordinates
# y = np.array([[...], [...], ...])
# R, d, rms = soder(x, y)

# Unit test for the challis function
def test_soder():
    """Unit test for the soder function using known transformation parameters."""
    print("SODER UNIT TEST")
    # Known transformation parameters
    R_known = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])  
    d_known = np.array([10.0, 0.0, 0.0])  # Translation vector

    x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

    x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.34, 1.28, 1.0],
            [-0.25, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.54, -1.12]
        ])

    y =  (R_known @ x.T).T + d_known  # Apply known transformation

    print("x:", x)
    print("y:", y)
    
    # Call the challis function
    R, d, rms = soder(x, y)
    print("R:\n", R)
    print("d:\n", d)
    print("rms:\n", rms)

# test_soder()
    # Assertions to check if the estimated parameters match the known ones
    # assert np.allclose(R, R_known, atol=1e-3), f"Rotation matrix mismatch: {R} vs {R_known}"
    # assert np.allclose(d, d_known, atol=1e-3), f"Translation vector mismatch: {d} vs {d_known}"
    # assert np.isclose(s, s_known, atol=1e-3), f"Scale factor mismatch: {s} vs {s_known}"
    # print(f"Test passed! RMS error: {rms:.6f}")

