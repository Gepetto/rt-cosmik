"""Small numerical helpers."""
import numpy as np
from numpy import linalg as LA
from scipy import signal

def trace(m):
    """Trace of the matrix ``m``, as a float."""
    return float(np.trace(m))


def norm(vector):
    """Euclidean norm of ``vector``."""
    return LA.norm(vector)

def col_vector_3D(a, b, c):
    """The column vector ``(a, b, c)``, as a (3, 1) float64 array."""
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)


def RMSE(est, ref):
    """Root mean square error of ``est`` against ``ref``, over their first axis.

    For (N, k) arrays, one value per column.
    """
    sq_err_sum=0
    for i in range(len(est)):
        sq_err_sum += pow(est[i] - ref[i], 2)
    
    rmse = np.sqrt(sq_err_sum/len(est))
    return rmse

                     


def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    """Zero-phase low-pass Butterworth filter of a whole recording.

    Filters forwards and backwards along the first axis (time), so the result
    has no delay; offline only, as it needs the whole signal.

    Args:
        data: samples along the first axis, any number of channels.
        cutoff_frequency: cutoff (Hz), below half the sampling frequency.
        order: filter order.
        sampling_frequency: sampling frequency (Hz).

    Returns:
        np.ndarray: the filtered data, the same shape as ``data``.
    """
    nyquist = 0.5 * sampling_frequency
    if not 0 < cutoff_frequency < nyquist:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    b, a = signal.butter(order, cutoff_frequency / nyquist, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


