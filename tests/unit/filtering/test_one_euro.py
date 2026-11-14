"""The 1-euro filter on marker frames, through MarkerFilter."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.filtering.iir import MarkerFilter


def one_euro(min_cutoff=1.0, beta=0.0, markers=2):
    settings = SimpleNamespace(fs=40, order=4, cutoff_freq=5, filter_type="one_euro",
                               one_euro_min_cutoff=min_cutoff, one_euro_beta=beta)
    return MarkerFilter(markers, settings)


def run(filt, frames):
    return np.stack([filt(f) for f in frames])


def test_first_frame_and_a_still_signal_pass_unchanged():
    frames = np.tile(np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]), (10, 1, 1))
    np.testing.assert_allclose(run(one_euro(), frames), frames)


def test_without_speed_term_it_is_a_first_order_low_pass():
    # 1 Hz at 40 Hz: alpha = dt / (dt + 1 / (2 pi fc)) = 0.1357
    frames = np.zeros((3, 2, 3))
    frames[1:] = 1.0
    out = run(one_euro(min_cutoff=1.0, beta=0.0), frames)
    np.testing.assert_allclose(out[1], 0.1357, atol=1e-4)
    np.testing.assert_allclose(out[2], 0.1357 + 0.1357 * (1 - 0.1357), atol=1e-4)


def test_output_does_not_depend_on_later_frames():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(30, 2, 3))
    np.testing.assert_array_equal(run(one_euro(beta=5.0), frames)[:20],
                                  run(one_euro(beta=5.0), frames[:20]))


def test_speed_term_cuts_the_lag_on_a_fast_ramp():
    frames = np.tile(np.arange(40.0)[:, None, None] * 0.05, (1, 2, 3))   # 2 m/s
    slow = run(one_euro(min_cutoff=1.0, beta=0.0), frames)[-1, 0, 0]
    fast = run(one_euro(min_cutoff=1.0, beta=10.0), frames)[-1, 0, 0]
    target = frames[-1, 0, 0]
    assert target - fast < 0.2 * (target - slow)


def test_each_coordinate_adapts_on_its_own():
    frames = np.zeros((40, 1, 3))
    frames[:, 0, 0] = np.arange(40.0) * 0.05
    frames[1:, 0, 1] = 1.0
    out = run(one_euro(min_cutoff=1.0, beta=10.0, markers=1), frames)
    np.testing.assert_allclose(out[2, 0, 2], 0.0)
    assert out[1, 0, 1] > 0.1357 + 1e-3      # the step itself raises that coordinate's cutoff
