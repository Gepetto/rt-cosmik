from dataclasses import dataclass, field
from scipy import signal
from typing import Union, Sequence, List
import numpy as np


class DimensionError(Exception):
    """Raise when the dimension of signals not as expected"""
    pass


@dataclass
class IIR:
    """
    IIR multi-channel filter
    """
    num_channel: int
    sampling_frequency: int

    raw_enabled: bool = field(repr=False, default=False)
    coeffs: List[tuple] = field(init=False, repr=False, default_factory=list)
    past_zi: List[np.ndarray] = field(init=False, repr=False, default=None)

    def _init_zi(self, signals: Union[List, np.ndarray]) -> None:
        """
        Initialize initial condition for first sample to reduce transient-state time of signals

        :param signals: Two dimensional matrix of [samples x channels]
        :type signals: List or numpy array
        """
        self.past_zi = list()
        first_sample = signals[..., 0]

        # Change dimension to match initial_zi dimension
        first_sample = np.expand_dims(
            np.expand_dims(first_sample, axis=-1), axis=0)

        for coeff in self.coeffs:
            initial_zi = signal.sosfilt_zi(coeff)
            initial_zi = np.repeat(np.expand_dims(
                initial_zi, axis=1), self.num_channel, axis=1)
            initial_zi *= first_sample
            self.past_zi.append(initial_zi)

    def set_raw_enabled(self, state: bool) -> None:
        self.raw_enabled = state
        self.past_zi = None

    def add_filter(self, order: int, cutoff: Union[Sequence, int, float], filter_type: str) -> None:
        """
        Add filter into cascading pipeline

        :param int order: An order of filter.
        :param Union[Sequence, int, float] cutoff: A critical frequency of the filter.
        :param str filter_type: Filter type can be 'lowpass', 'highpass', 'bandstop' and 'bandpass'.
        """

        new_filter_coeff = signal.butter(
            order, cutoff, filter_type, output='sos', fs=self.sampling_frequency)

        self.coeffs.append(new_filter_coeff)

    def add_sos(self, sos: np.ndarray) -> None:
        """
        Add sos filter into cascading pipeline

        :param ndarray sos: A filter coefficient.
        """
        self.coeffs.append(sos)

    def filter(self, raw_signal: Union[List, np.ndarray]) -> np.ndarray:
        """
        Filter a sequence of multi-channel samples

        :param raw_signal: Two dimensional matrix of [samples x channels]
        :type raw_signal: Union[List, np.ndarray]
        :return np.ndarray
        """
        # Check if input is List or numpy array
        if isinstance(raw_signal, list):
            filt_signal = np.array(list(zip(*raw_signal)))
        elif isinstance(raw_signal, np.ndarray):
            filt_signal = raw_signal.T

        # If raw_mode then return
        if self.raw_enabled:
            return filt_signal.T

        # Check input correctness
        signal_dim = filt_signal.shape
        if len(signal_dim) != 2:
            raise DimensionError(f'Input signal dimension must be equal to 2')
        if signal_dim[0] != self.num_channel:
            raise DimensionError(
                f'Number of channels must be equal to {self.num_channel}')

        if self.past_zi is None:
            self._init_zi(filt_signal)

        for index, (sos, past_zi) in enumerate(zip(self.coeffs, self.past_zi)):
            filt_signal, zi = signal.sosfilt(sos, filt_signal, zi=past_zi)
            self.past_zi[index] = zi

        filtered = filt_signal.T
        return filtered

class MarkerFilter:
    """Low-pass a stream of marker frames, one frame at a time.

    :class:`IIR` is stateful -- every call advances ``past_zi`` by as many
    samples as it is handed -- so it must be fed exactly one new sample per
    frame. Handing it a buffer of the last N frames instead, as this pipeline
    once did, re-feeds N-1 samples it has already consumed and advances its
    state N steps per real sample. That is not the filter the settings describe:
    at N = 7 the gain at 3 Hz was 0.78 where a true 4th-order 5 Hz Butterworth
    gives 0.99, and the response moved non-monotonically with N (0.97, 0.88,
    0.78, 1.02 at N = 3, 5, 7, 10) -- a response that depends on the horizon
    length is not a response at all.

    Driven one frame at a time, ``settings.order`` and ``settings.cutoff_freq``
    mean what they say.

    No frame buffer is needed here: ``HumanSolver.solve`` takes a single frame
    and the moving-horizon IK keeps its own window internally.
    """

    def __init__(self, marker_count, settings):
        self.marker_count = marker_count
        self.channels = 3 * marker_count
        self.iir = IIR(num_channel=self.channels,
                       sampling_frequency=settings.fs)
        self.iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                            filter_type=settings.filter_type)

    def __call__(self, frame):
        """One ``(marker_count, 3)`` frame in, one filtered frame out.

        The first call seeds the filter state from that frame, so the output
        starts at the signal rather than ramping up from zero.
        """
        flat = np.asarray(frame, dtype=float).reshape(1, self.channels)
        return self.iir.filter(flat).reshape(self.marker_count, 3)
