"""The 1-euro filter (Casiez et al., CHI 2012): a low-pass whose cutoff rises with speed."""
import numpy as np

DERIVATIVE_CUTOFF_HZ = 1.0


def _alpha(cutoff, dt):
    return dt / (dt + 1.0 / (2.0 * np.pi * cutoff))


class OneEuro:
    """Filters any array, one sample per call, each element on its own."""

    def __init__(self, fs, min_cutoff, beta):
        self.dt, self.min_cutoff, self.beta = 1.0 / fs, min_cutoff, beta
        self.x = self.dx = None

    def __call__(self, sample):
        sample = np.asarray(sample, dtype=float)
        if self.x is None:
            self.x, self.dx = sample.copy(), np.zeros_like(sample)
            return sample
        a = _alpha(DERIVATIVE_CUTOFF_HZ, self.dt)
        self.dx = a * (sample - self.x) / self.dt + (1 - a) * self.dx
        a = _alpha(self.min_cutoff + self.beta * np.abs(self.dx), self.dt)
        self.x = a * sample + (1 - a) * self.x
        return self.x.copy()
