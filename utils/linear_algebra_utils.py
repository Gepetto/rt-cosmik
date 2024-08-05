import numpy as np
import pathlib as pl
import os
from numpy import linalg as LA
import scipy as sc
from scipy import linalg as linalg_scipy
import scipy.optimize as opt
import math as m
from scipy import interpolate
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation 
from scipy.spatial.transform import Rotation as R

def trace(m):
    tr = 0
    for i in range(len(m)):
        tr = tr+m[i][i]
    return float(tr)

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=object)

def rotmat_from_values(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.array([[x0, x1, x2], 
                     [y0, y1, y2], 
                     [z0, z1, z2]], dtype=object).astype(float)

def rotmat_from_row_vecs(x, y, z):
    return np.array([[x[0], x[1], x[2]], 
                     [y[0], y[1], y[2]], 
                     [z[0], z[1], z[2]]], dtype=object).astype(float)

def identity_3D():
    return np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]], dtype=object).astype(float)

def norm(vector):
    return LA.norm(vector)

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=object)

def row_vector_3D(x):
    return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=object)

def row_vector_3D(a, b, c):
    return np.array([float(a), float(b), float(c)], dtype=object)

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
    
    rmse = m.sqrt(sq_err_sum/len(est))
    return rmse

def vec_to_skewmat(x):
    return np.array([[0.0, -x[2], x[1]], 
                     [x[2], 0.0, -x[0]], 
                     [-x[1], x[0], 0.0]], dtype=object).astype(float)
                     
def skewmat_to_vec(skew):
    return col_vector_3D(-skew[1][2], skew[0][2], -skew[0][1])

def rot_to_cayley(rot):
    cayley_skew = (identity_3D() - rot)*np.linalg.inv((identity_3D() + rot))
    return skewmat_to_vec(cayley_skew)

def cayley_to_rot(cayley):
    return (identity_3D() - vec_to_skewmat(cayley))*np.linalg.inv((identity_3D() + vec_to_skewmat(cayley)))


def make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

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