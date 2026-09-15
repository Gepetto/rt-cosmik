"""ContactVision foot contact detector, run causally on a live keypoint stream.

Model from https://github.com/DaeeYong/ContactVision (commit 2b27a18), used as released.
MIT License, Copyright (c) 2026 DaeYong Kim. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the following conditions: The above
copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
WARRANTY OF ANY KIND.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn


class FootContactTransformer(nn.Module):
    """ContactVision network: 13 keypoints x (x, y, confidence) -> 4 logits per frame."""

    def __init__(self, input_dim, embed_dim, n_heads, ff_dim, num_layers, dropout):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, 10000, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_fc(x)
        x = x + self.pos_enc[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.classifier(x)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = state["hyperparams"]
    model = FootContactTransformer(39, hp["embed_dim"], hp["n_heads"], hp["ff_dim"],
                                   hp["num_layers"], hp["dropout"])
    model.load_state_dict(state["model_state"])
    return model.eval().to(device)


def input_features(keypoints, inside, image_height):
    """One camera frame as model input (39,): mid-hip-relative 1080p pixels + confidence.

    NLF gives no keypoint confidence: 0.9 inside the image, 0 outside.
    """
    relative = (keypoints - keypoints[:1]) * (1080.0 / image_height)
    return np.concatenate([np.nan_to_num(relative), 0.9 * inside[:, None]],
                          axis=1).reshape(39).astype(np.float32)


class ContactVisionStream:
    """Per-camera 30 Hz input windows and camera-fused logits at recent times.

    On CUDA the forward pass is a CUDA graph: fewer kernel launches means fewer GIL
    releases, which a busy viewer thread otherwise turns into multi-ms stalls.
    """

    FPS = 30.0
    WINDOW = 128

    def __init__(self, model, num_cameras, device):
        self.model = model
        self.device = device
        self.num_cameras = num_cameras
        self._inputs = np.zeros((num_cameras, self.WINDOW, 39), dtype=np.float32)
        self._visible = np.zeros((num_cameras, self.WINDOW))
        self._ticks = None          # sample times, oldest first
        self._last = None           # (t, inputs, visible) of the previous frame
        self._logits = None         # (C, WINDOW, 4)
        self._stale = True
        self._graph = None
        if str(device).startswith("cuda") and isinstance(model, nn.Module):
            self._capture()

    def _capture(self):
        shape = (self.num_cameras, self.WINDOW, 39)
        self._graph_input = torch.zeros(shape, device=self.device)
        self._host_input = torch.zeros(shape).pin_memory()
        with torch.no_grad():
            side = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.model(self._graph_input)
            torch.cuda.current_stream(self.device).wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._graph_output = self.model(self._graph_input)

    def push(self, t, inputs, visible):
        """Add one frame: inputs (C, 39), visible (C,) in [0, 1], at time t (s)."""
        inputs = np.asarray(inputs, dtype=np.float32)
        visible = np.asarray(visible, dtype=float)
        if self._ticks is None:
            # Full window from the start: the positional encoding expects 128 samples.
            self._inputs[:] = inputs[:, None, :]
            self._visible[:] = visible[:, None]
            self._ticks = t - np.arange(self.WINDOW - 1, -1, -1) / self.FPS
            self._last = (t, inputs, visible)
            self._stale = True
            return
        t_prev, inputs_prev, visible_prev = self._last
        if t <= t_prev:
            return
        tick = self._ticks[-1] + 1.0 / self.FPS
        while tick <= t + 1e-9:
            w = (tick - t_prev) / (t - t_prev)
            self._inputs[:, :-1] = self._inputs[:, 1:]
            self._inputs[:, -1] = (1.0 - w) * inputs_prev + w * inputs
            self._visible[:, :-1] = self._visible[:, 1:]
            self._visible[:, -1] = (1.0 - w) * visible_prev + w * visible
            self._ticks[:-1] = self._ticks[1:]
            self._ticks[-1] = tick
            self._stale = True
            tick += 1.0 / self.FPS
        self._last = (t, inputs, visible)

    @torch.inference_mode()
    def _run(self):
        if self._graph is not None:
            self._host_input.numpy()[:] = self._inputs
            self._graph_input.copy_(self._host_input, non_blocking=True)
            self._graph.replay()
            self._logits = self._graph_output.float().cpu().numpy()
        else:
            x = torch.from_numpy(self._inputs).to(self.device)
            self._logits = self.model(x).float().cpu().numpy()
        self._stale = False

    def logits(self, times):
        """Logits at times (T, 4), averaged over cameras seeing the feet; NaN if none."""
        times = np.asarray(times, dtype=float)
        if self._ticks is None:
            return np.full((len(times), 4), np.nan)
        if self._stale:
            self._run()
        per_camera = np.full((self.num_cameras, len(times), 4), np.nan)
        for c in range(self.num_cameras):
            seen = np.interp(times, self._ticks, self._visible[c]) >= 0.5
            if not seen.any():
                continue
            for j in range(4):
                per_camera[c, :, j] = np.interp(times, self._ticks, self._logits[c, :, j])
            per_camera[c, ~seen] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(per_camera, axis=0)
