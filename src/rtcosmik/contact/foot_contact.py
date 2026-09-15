"""Per-node foot contact probability for the MHE window.

probability = sigmoid(ContactVision logit) x 3D gate (foot low and slow). The gate
reads the filtered markers and the keypoints go through the same filter, so every
node's probability is on the solver's timeline.
"""

from collections import deque

import numpy as np

from rtcosmik.contact import points
from rtcosmik.contact.contactvision import ContactVisionStream, input_features, load_model
from rtcosmik.filtering.iir import IIR

MAX_HEIGHT = 0.08          # m above standing height
HEIGHT_SOFTNESS = 0.015    # m
MAX_SPEED = 0.8            # m/s
SPEED_SOFTNESS = 0.15      # m/s
CALIBRATION_FRAMES = 10    # frames defining each point's standing height


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FootContact:
    """Call update() once per frame given to the MHE, calibration frame included."""

    def __init__(self, settings, num_cameras, image_size, logger=None):
        self.settings = settings
        self.num_cameras = num_cameras
        self.width, self.height = image_size
        self.extra_canonical_points = points.extra_canonical_points()
        self._lower = points.lower_body_indices(settings.marker_names,
                                                settings.nlf_indices)
        self._feet = [list(settings.marker_names).index(name)
                      for name in points.PROBABILITY_MARKERS]
        self._stream = ContactVisionStream(
            load_model(settings.contactvision_path, settings.device),
            num_cameras, settings.device)

        self._held = [None] * num_cameras       # last keypoints showing both feet
        self._filters = [None] * num_cameras

        self._calibration = []
        self._floor = None
        self._previous = None                   # (t, feet)
        self._times = deque(maxlen=settings.N)
        self._gates = deque(maxlen=settings.N)
        self.latest = np.zeros(len(points.PROBABILITY_MARKERS))

    def update(self, t, keypoints, markers):
        """Add a frame; return (N, 4) probabilities, oldest node first.

        Args:
            t: frame time (s).
            keypoints: per camera, NLF (J, 2) pixels incl. extra points, or None.
            markers: (M, 3) filtered world markers, settings.marker_names order.
        """
        self._push_keypoints(t, keypoints)
        gate = self._gate(t, np.asarray(markers, dtype=float)[self._feet])
        if not self._times:
            self._times.extend([t] * self.settings.N)
            self._gates.extend([gate] * self.settings.N)
        else:
            self._times.append(t)
            self._gates.append(gate)

        logits = self._stream.logits(list(self._times))
        probability = np.where(np.isfinite(logits), _sigmoid(np.nan_to_num(logits)), 0.0)
        probability *= np.array(self._gates)
        self.latest = probability[-1]
        return probability

    def _push_keypoints(self, t, keypoints):
        inputs = np.zeros((self.num_cameras, 39), dtype=np.float32)
        visible = np.zeros(self.num_cameras)
        for c in range(self.num_cameras):
            current = None
            if keypoints[c] is not None:
                current = np.asarray(keypoints[c], dtype=float)[self._lower]
            inside = np.zeros(len(points.LOWER_BODY), dtype=bool)
            if current is not None:
                inside = (np.isfinite(current).all(axis=1)
                          & (current[:, 0] >= 0) & (current[:, 0] < self.width)
                          & (current[:, 1] >= 0) & (current[:, 1] < self.height))
            feet_seen = bool(inside[7:].all())
            if feet_seen:
                self._held[c] = current
            if self._held[c] is None:
                continue          # never saw the feet: does not vote
            # Hold the last good keypoints so the filter never jumps.
            raw = np.append(input_features(self._held[c], inside, self.height),
                            float(feet_seen))
            if self._filters[c] is None:
                self._filters[c] = IIR(num_channel=raw.size,
                                       sampling_frequency=self.settings.fs)
                self._filters[c].add_filter(order=self.settings.order,
                                            cutoff=self.settings.cutoff_freq,
                                            filter_type=self.settings.filter_type)
            filtered = self._filters[c].filter(raw[None, :])[-1]
            inputs[c], visible[c] = filtered[:39], filtered[39]
        self._stream.push(t, inputs, visible)

    def _gate(self, t, feet):
        speed = np.zeros(len(feet))
        if self._previous is not None and t > self._previous[0]:
            speed = np.linalg.norm(feet - self._previous[1], axis=1) / (t - self._previous[0])
        self._previous = (t, feet)

        if self._floor is None:
            self._calibration.append(feet[:, 2])
            if len(self._calibration) >= CALIBRATION_FRAMES:
                self._floor = np.median(self._calibration, axis=0)
            return np.zeros(len(feet))

        height = feet[:, 2] - self._floor
        return (_sigmoid((MAX_HEIGHT - height) / HEIGHT_SOFTNESS)
                * _sigmoid((MAX_SPEED - speed) / SPEED_SOFTNESS))
