"""ContactVision streaming, with stand-in models whose output is known."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from rtcosmik.config_loader import settings                       # noqa: E402
from rtcosmik.contact.contactvision import (                      # noqa: E402
    ContactVisionStream, input_features, load_model)


def echo_model(x):
    """Logit j = input feature j: predictions become the resampled inputs."""
    return x[..., :4]


def test_input_features_are_hip_relative_1080p_with_confidence():
    keypoints = np.zeros((13, 2))
    keypoints[0] = (100.0, 200.0)                  # mid-hip
    keypoints[9] = (110.0, 260.0)                  # left heel
    inside = np.ones(13, dtype=bool)
    inside[12] = False
    features = input_features(keypoints, inside, image_height=720).reshape(13, 3)
    np.testing.assert_allclose(features[0], (0.0, 0.0, 0.9))
    np.testing.assert_allclose(features[9], (15.0, 90.0, 0.9))     # x1.5 to 1080p
    assert features[12, 2] == 0.0


def test_input_features_have_no_nan():
    keypoints = np.full((13, 2), np.nan)
    features = input_features(keypoints, np.zeros(13, dtype=bool), 720)
    assert np.isfinite(features).all()


def _signal(t):
    return np.stack([np.sin(2 * np.pi * 0.7 * t), np.cos(2 * np.pi * 1.3 * t),
                     0.5 * t, np.sin(t)], axis=-1)


def test_resampling_matches_linear_interpolation_at_30hz():
    stream = ContactVisionStream(echo_model, num_cameras=1, device="cpu")
    times = np.arange(200) / 40.0
    values = _signal(times)
    for t, v in zip(times, values):
        inputs = np.zeros((1, 39), dtype=np.float32)
        inputs[0, :4] = v
        stream.push(t, inputs, [1.0])
    ticks = stream._ticks
    assert np.allclose(np.diff(ticks), 1.0 / 30.0)
    assert ticks[-1] <= times[-1] + 1e-9 < ticks[-1] + 1.0 / 30.0
    for j in range(4):
        expected = np.interp(ticks, times, values[:, j])
        np.testing.assert_allclose(stream._inputs[0, :, j], expected, atol=1e-5)


def test_logits_are_read_back_at_the_requested_times():
    stream = ContactVisionStream(echo_model, num_cameras=1, device="cpu")
    times = np.arange(200) / 40.0
    for t, v in zip(times, _signal(times)):
        inputs = np.zeros((1, 39), dtype=np.float32)
        inputs[0, :4] = v
        stream.push(t, inputs, [1.0])
    query = times[-10:-1]            # a window of recent frames, as the MHE asks
    logits = stream.logits(query)
    # A ramp survives linear interpolation exactly, so any time offset shows:
    # one frame late would be 0.0125 off.
    np.testing.assert_allclose(logits[:, 2], _signal(query)[:, 2], atol=1e-5)
    # Curved signals are smoothed a little by interpolating twice (40 -> 30 -> 40 Hz).
    np.testing.assert_allclose(logits, _signal(query), atol=0.02)


def test_only_cameras_seeing_the_feet_vote():
    stream = ContactVisionStream(echo_model, num_cameras=2, device="cpu")
    for k in range(60):
        inputs = np.zeros((2, 39), dtype=np.float32)
        inputs[0, :4] = 1.0
        inputs[1, :4] = 5.0
        stream.push(k / 40.0, inputs, [1.0, 0.0])
    np.testing.assert_allclose(stream.logits([59 / 40.0]), [[1.0] * 4])


def test_no_camera_seeing_the_feet_gives_nan():
    stream = ContactVisionStream(echo_model, num_cameras=2, device="cpu")
    for k in range(10):
        stream.push(k / 40.0, np.ones((2, 39), dtype=np.float32), [0.0, 0.0])
    assert np.isnan(stream.logits([9 / 40.0])).all()


def test_the_model_runs_once_per_new_30hz_sample():
    calls = []

    def counting_model(x):
        calls.append(1)
        return x[..., :4]

    stream = ContactVisionStream(counting_model, num_cameras=1, device="cpu")
    for k in range(80):                           # 2 s at 40 fps
        stream.push(k / 40.0, np.zeros((1, 39), dtype=np.float32), [1.0])
        stream.logits([k / 40.0])
    assert len(calls) == 1 + 59                   # first frame, then 30 Hz samples


def test_cuda_graph_predicts_exactly_what_the_model_does():
    """The graphed forward is only a faster launch of the same computation."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = torch.nn.Sequential(torch.nn.Linear(39, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).cuda().eval()
    graphed = ContactVisionStream(model, num_cameras=3, device="cuda")
    eager = ContactVisionStream(model, num_cameras=3, device="cuda")
    eager._graph = None
    assert graphed._graph is not None
    rng = np.random.default_rng(0)
    for k in range(80):
        inputs = rng.normal(size=(3, 39)).astype(np.float32)
        graphed.push(k / 40.0, inputs, [1.0, 1.0, 1.0])
        eager.push(k / 40.0, inputs, [1.0, 1.0, 1.0])
        times = [k / 40.0]
        np.testing.assert_array_equal(graphed.logits(times), eager.logits(times))


def test_released_checkpoint_predicts_four_logits_per_frame():
    if not os.path.isfile(settings.contactvision_path):
        pytest.skip("ContactVision checkpoint not fetched (scripts/bash/fetch_models.sh)")
    model = load_model(settings.contactvision_path, "cpu")
    with torch.inference_mode():
        out = model(torch.zeros(2, ContactVisionStream.WINDOW, 39))
    assert out.shape == (2, ContactVisionStream.WINDOW, 4)
    assert torch.isfinite(out).all()
