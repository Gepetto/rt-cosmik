import numpy as np
from numpy import linalg as LA
from scipy import signal

def trace(m):
    return float(np.trace(m))


def norm(vector):
    return LA.norm(vector)

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)


def RMSE(est, ref):
    sq_err_sum=0
    for i in range(len(est)):
        sq_err_sum += pow(est[i] - ref[i], 2)
    
    rmse = np.sqrt(sq_err_sum/len(est))
    return rmse

                     


def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    if not 0 < cutoff_frequency < nyquist:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    b, a = signal.butter(order, cutoff_frequency / nyquist, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


