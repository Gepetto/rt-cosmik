import numpy as np
from numpy import linalg as LA
from scipy import signal

def trace(m):
    return float(np.trace(m))

def rotmat_from_values(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.array([[x0, x1, x2], 
                     [y0, y1, y2], 
                     [z0, z1, z2]], dtype=np.float64)

def rotmat_from_row_vecs(x, y, z):
    return np.array([[x[0], x[1], x[2]], 
                     [y[0], y[1], y[2]], 
                     [z[0], z[1], z[2]]], dtype=np.float64)

def identity_3D():
    return np.eye(3, dtype=np.float64)

def norm(vector):
    return LA.norm(vector)

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)

def row_vector_3D(a, b, c):
    return np.array([float(a), float(b), float(c)], dtype=np.float64)

def col_vector_3D_from_tab(x):
    return np.array([[float(x[0])], [float(x[1])], [float(x[2])]])

def rotmat(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.array([[x0, x1, x2], 
                     [y0, y1, y2], 
                     [z0, z1, z2]]).astype(float)

def RMSE(est, ref):
    sq_err_sum=0
    for i in range(len(est)):
        sq_err_sum += pow(est[i] - ref[i], 2)
    
    rmse = np.sqrt(sq_err_sum/len(est))
    return rmse

def vec_to_skewmat(x):
    return np.array([[0.0, -x[2], x[1]], 
                     [x[2], 0.0, -x[0]], 
                     [-x[1], x[0], 0.0]], dtype=object).astype(float)
                     
def skewmat_to_vec(skew):
    return col_vector_3D(-skew[1][2], skew[0][2], -skew[0][1])

def rot_to_cayley(rot):
    """Convert rotation matrix to Cayley representation."""
    cayley_skew = (identity_3D() - rot) @ np.linalg.inv(identity_3D() + rot)
    return skewmat_to_vec(cayley_skew)

def cayley_to_rot(cayley):
    return (identity_3D() - vec_to_skewmat(cayley))*np.linalg.inv((identity_3D() + vec_to_skewmat(cayley)))

def make_homogeneous_rep_matrix(R, t):
    if t.shape not in [(3,), (3, 1)]:
        raise ValueError("Translation vector must be of shape (3,) or (3,1)")
    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    return P

def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    if not 0 < cutoff_frequency < nyquist:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    b, a = signal.butter(order, cutoff_frequency / nyquist, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def low_pass_filter_data(data,nbutter=5):
    '''This function filters and elaborates data used in the identification process. 
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)'''
    
    b, a = signal.butter(nbutter, 0.01*5 / 2, "low")
   
    #data = signal.medfilt(data, 3)
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    # nbord = 5 * nbutter
    # data = np.delete(data, np.s_[0:nbord], axis=0)
    # data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data