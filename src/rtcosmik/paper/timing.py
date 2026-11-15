"""Timing of a paced replay and of recorded camera timestamps."""
import numpy as np


def statistics(samples):
    """Median, 95th and 99th percentiles and maximum of ``samples``."""
    samples = np.asarray(samples, dtype=float)
    return {"median": float(np.median(samples)), "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)), "max": float(np.max(samples))}


def paced_latency(processing_ms, period_ms):
    """Delay from each frame's arrival to its result, frames arriving every ``period_ms``.

    A frame is processed when it has arrived and the previous one is done.
    """
    latency, done = [], 0.0
    for index, duration in enumerate(processing_ms):
        arrival = index * period_ms
        done = max(arrival, done) + duration
        latency.append(done - arrival)
    return np.array(latency)


def missed_deadlines(latency_ms, period_ms):
    """Frames whose result was not ready when the next frame arrived."""
    return int(np.sum(np.asarray(latency_ms) > period_ms))


def timestamp_report(times, period_s):
    """Frame deficit, long intervals and inter-camera spread of ``times`` (F, C), in seconds.

    ``missing_frames`` is how many more frames the nominal rate gives over the recording's
    duration; ``long_intervals`` counts the gaps of more than one and a half periods.
    """
    times = np.asarray(times, dtype=float)
    intervals = np.diff(times, axis=0)
    expected = np.rint((times[-1] - times[0]) / period_s)
    spread = times.max(axis=1) - times.min(axis=1)
    return {"frames": int(len(times)),
            "missing_frames": [int(n) for n in expected - (len(times) - 1)],
            "long_intervals": [int(n) for n in (intervals > 1.5 * period_s).sum(axis=0)],
            "interval_ms": [dict(statistics(1e3 * intervals[:, camera]),
                                 mean=float(1e3 * intervals[:, camera].mean()))
                            for camera in range(times.shape[1])],
            "spread_ms": statistics(1e3 * spread),
            "offset_to_first_ms": [float(v) for v in 1e3 * np.median(times - times[:, :1], axis=0)]}
