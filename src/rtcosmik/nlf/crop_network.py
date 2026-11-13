"""NLF's crop network in exportable form, its ONNX export and its TensorRT engine.

Recipe and validation: ``docs/research/nlf-tensorrt/``.
"""
import hashlib
import importlib.util
import json
import os
import sys
import types
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

MAX_CROPS = 8
REBUILD = "rebuild it with scripts/python/core/export_nlf_engine.py"
_TRT_LOGGER = None


def _trt_logger():
    global _TRT_LOGGER
    import tensorrt as trt
    if _TRT_LOGGER is None:
        _TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    return _TRT_LOGGER


def _efficientnet(centered_stride):
    """NLF's own EfficientNetV2 port, with its two training-stack imports shimmed."""
    florch = types.ModuleType("florch")
    florch.layers = types.ModuleType("florch.layers")
    florch.layers.BatchRenorm2d = torch.nn.BatchNorm2d
    spu = types.ModuleType("simplepyutils")
    spu.FLAGS = types.SimpleNamespace(centered_stride=centered_stride)
    shims = {"florch": florch, "florch.layers": florch.layers, "simplepyutils": spu}
    saved = {k: sys.modules.get(k) for k in shims}
    sys.modules.update(shims)
    try:
        spec = importlib.util.spec_from_file_location(
            "nlf_efficientnet", Path(__file__).parent / "third_party" / "nlf_efficientnet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def _backbone(release):
    crop = release.crop_model
    eff = _efficientnet(bool(crop.heatmap_head.centered_stride))
    size = "s" if sum(p.numel() for p in crop.backbone.parameters()) < 50e6 else "l"
    features = getattr(eff, f"efficientnet_v2_{size}")(
        norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3), weights=None).features
    state = crop.backbone.state_dict()
    features.load_state_dict({k[2:]: v for k, v in state.items() if k.startswith("1.")},
                             strict=True)
    return features.eval()


def _head_layer(release):
    state = release.crop_model.heatmap_head.layer.state_dict()
    conv = state["0.weight"]
    layer = torch.nn.Sequential(torch.nn.Conv2d(conv.shape[1], conv.shape[0], 1, bias=False),
                                torch.nn.BatchNorm2d(conv.shape[0], eps=1e-3), torch.nn.SiLU())
    layer.load_state_dict(state, strict=True)
    return layer.eval()


def load_release(path):
    """The released TorchScript model, on the CPU in fp32."""
    import torchvision  # noqa: F401  registers torchvision::nms
    return torch.jit.load(str(path), map_location="cpu").float().eval()


def inv3x3(m):
    """Closed-form inverse of (..., 3, 3) matrices."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    co = torch.stack((e * i - f * h, c * h - b * i, b * f - c * e,
                      f * g - d * i, a * i - c * g, c * d - a * f,
                      d * h - e * g, b * g - a * h, a * e - b * d), -1).unflatten(-1, (3, 3))
    det = a * co[..., 0, 0] + b * co[..., 1, 0] + c * co[..., 2, 0]
    return co / det[..., None, None]


def _masked_mean(x, valid):
    """Mean of (B, P, 2) over the valid landmarks, 0 if none."""
    v = valid.to(x.dtype).unsqueeze(-1)
    return (x * v).sum((1, 2), keepdim=True) / torch.clamp(v.sum((1, 2), keepdim=True), min=1.0)


class NLFCrop(torch.nn.Module):
    """Crop and crop intrinsics in, landmarks (mm, crop camera frame) and uncertainties out.

    ``half`` runs the backbone and head layer in fp16; the localizer convolution
    and everything after it stay in fp32.
    """

    def __init__(self, release, cano, half=False):
        super().__init__()
        head = release.crop_model.heatmap_head
        self.backbone = _backbone(release)
        self.layer = _head_layer(release)
        self.half_features = half
        if half:
            self.backbone.half()
            self.layer.half()
        self.proc_side, self.stride = int(head.proc_side), int(head.stride_test)
        self.centered = bool(head.centered_stride)
        self.box_size_m, self.depth = float(head.box_size_m), int(head.depth)
        self.uncert_bias, self.uncert_bias2 = float(head.uncert_bias), float(head.uncert_bias2)
        self.uncert_factor = 1.0 if bool(head.fix_uncert_factor) else 3.0
        with torch.no_grad():
            w = head.weight_field(cano.float())
            w = w.unflatten(-1, (int(head.backbone_link_dim) + 1, 2 + self.depth))
        self.register_buffer(
            "w", w[:, :-1, :].permute(0, 2, 1).flatten(0, 1)[..., None, None].contiguous())
        self.register_buffer("b", w[:, -1, :].reshape(-1).contiguous())

    def _to_image(self, c):
        last = self.proc_side - 1
        out = c * (last - last % self.stride)
        return out + self.stride // 2 if self.centered else out

    @staticmethod
    def _decode(heatmap, dims):
        """Expected coordinate in [0, 1] along each of ``dims``, in the order given."""
        out = []
        for d in dims:
            marginal = heatmap.sum([x for x in dims if x != d], keepdim=True)
            n = heatmap.shape[d]
            coords = torch.arange(n, dtype=torch.float32, device=heatmap.device) / (n - 1)
            shape = [n if i == d else 1 for i in range(heatmap.dim())]
            out.append((marginal * coords.reshape(shape)).sum(d, keepdim=True).flatten(2))
        return torch.cat(out, -1)

    def _in_fov(self, coords, border):
        offset = 0.0 if self.centered else -self.stride / 2.0
        lo = self.stride * border + offset
        hi = self.proc_side - self.stride * border + offset
        return ((coords >= lo) & (coords <= hi)).all(-1)

    def forward(self, image, intrinsics, landmarks):
        x = 2 * image - 1
        if self.half_features:
            x = x.half()
        feats = self.layer(self.backbone(x)).float()
        logits = F.conv2d(feats, self.w, self.b)
        n, h, wd = logits.shape[0], logits.shape[2], logits.shape[3]
        logits = logits.reshape(n, -1, 2 + self.depth, h, wd).index_select(1, landmarks)
        unc_map = logits[:, :, 0]
        hm2 = torch.softmax(logits[:, :, 1].flatten(2), -1).reshape(n, -1, h, wd)
        xy_metric = self._decode(hm2, [3, 2])
        hm25 = torch.softmax(logits[:, :, 2:].flatten(2), -1).reshape(n, -1, self.depth, h, wd)
        coords25 = self._decode(hm25, [4, 3, 2])
        unc = F.softplus((unc_map * hm25.sum(2)).sum((2, 3)) + self.uncert_bias) + self.uncert_bias2

        coords2d = self._to_image(coords25[..., :2])
        rel = torch.cat((self._to_image(xy_metric) * self.box_size_m / self.proc_side,
                         coords25[..., 2:] * self.box_size_m), -1)
        points = self._absolute(coords2d, rel, intrinsics, unc < 0.3) * 1000
        return points, unc * self.uncert_factor

    def _absolute(self, coords2d, rel, k, valid):
        """NLF's ``reconstruct_absolute``: full perspective, mix 0.5, border factors 1.0 / 0.6."""
        ones = torch.ones_like(coords2d[..., :1])
        norm2d = (torch.cat((coords2d, ones), -1) @ inv3x3(k).transpose(1, 2))[..., :2]
        valid1 = self._in_fov(coords2d, 1.0) & valid

        s2 = torch.sqrt(_masked_mean(norm2d.square(), valid1) + 1e-10)
        a = norm2d / s2
        back = norm2d * rel[:, :, 2:] - rel[:, :, :2]
        sb = torch.sqrt(_masked_mean(back.square(), valid1) + 1e-10)
        rhs = (back / sb).flatten(1).unsqueeze(-1)                       # (B, 2P, 1)
        one, zero = torch.ones_like(a[..., 0]), torch.zeros_like(a[..., 0])
        A = torch.stack((torch.stack((one, zero, -a[..., 0]), -1),
                         torch.stack((zero, one, -a[..., 1]), -1)), 2).flatten(1, 2)
        weights = (valid1.to(a.dtype) + 1e-8).unsqueeze(-1).expand_as(a).flatten(1)
        wA = weights.unsqueeze(-1) * A
        gram = wA.transpose(1, 2) @ A + 1e-4 * torch.eye(3, dtype=A.dtype, device=A.device)
        ref = inv3x3(gram) @ (wA.transpose(1, 2) @ rhs)
        ref = torch.cat((ref[:, :2] * sb, ref[:, 2:] * (sb / s2)), 1).squeeze(-1)

        abs3d = rel + ref.unsqueeze(1)
        abs2d = torch.cat((norm2d, ones), -1) * (rel[..., 2] + ref[:, 2:3]).unsqueeze(-1)
        abs2d = 0.5 * abs3d + 0.5 * abs2d
        proj = abs3d / torch.clamp(abs3d[..., 2:], min=0.1)
        proj = torch.einsum("bnk,bjk->bnj", proj, k[..., :2, :])
        valid2 = self._in_fov(proj, 0.6) & (abs3d[..., 2] > 0.001)
        return torch.where(valid2.unsqueeze(-1), abs2d, abs3d)


def landmark_rows(engine_vertices, indices):
    """Row of each of ``indices`` in an engine built for ``engine_vertices``."""
    row = {int(v): i for i, v in enumerate(engine_vertices)}
    missing = sorted({int(v) for v in indices} - set(row))
    if missing:
        raise KeyError(f"NLF engine holds no localizer weights for vertices {missing}; {REBUILD}")
    return [row[int(v)] for v in indices]


def meta_path(engine_path):
    return Path(f"{engine_path}.json")


def _write(path, data):
    tmp = Path(f"{path}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_engine(engine_path, serialized, meta):
    """Write the engine, then its sidecar naming the engine's sha256, each atomically."""
    data = bytes(serialized)
    _write(engine_path, data)
    digest = hashlib.sha256(data).hexdigest()
    _write(meta_path(engine_path), json.dumps({**meta, "engine_sha256": digest}).encode())


def checked_meta(engine_path, data, tensorrt_version):
    """Sidecar of the engine file ``data``, refused if it is another engine's or another TensorRT's."""
    path = meta_path(engine_path)
    if not path.is_file():
        raise RuntimeError(f"NLF engine {engine_path} has no sidecar {path.name}; {REBUILD}")
    meta = json.loads(path.read_text())
    if meta.get("engine_sha256") != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"{path} is not the sidecar of this engine file; {REBUILD}")
    if meta["tensorrt"] != tensorrt_version:
        raise RuntimeError(f"NLF engine {engine_path} was built with TensorRT {meta['tensorrt']}, "
                           f"{tensorrt_version} is installed; {REBUILD}")
    return meta


def export_onnx(release, cano, onnx_path, half, device="cuda:0"):
    """Write the crop network for the vertices ``cano`` to ONNX, batch and landmark count dynamic."""
    model = NLFCrop(release, cano, half=half).eval().to(device)
    n = cano.shape[0]
    args = (torch.zeros(2, 3, model.proc_side, model.proc_side, device=device),
            torch.eye(3, device=device).repeat(2, 1, 1),
            torch.arange(min(n, 3), device=device))
    with torch.no_grad():
        torch.onnx.export(
            model, args, str(onnx_path), opset_version=17, dynamo=False,
            input_names=["image", "intrinsics", "landmarks"],
            output_names=["points_mm", "uncertainty"],
            dynamic_axes={"image": {0: "crops"}, "intrinsics": {0: "crops"},
                          "landmarks": {0: "landmarks"},
                          "points_mm": {0: "crops", 1: "landmarks"},
                          "uncertainty": {0: "crops", 1: "landmarks"}})
    return model


def build_engine(onnx_path, engine_path, vertices, half, side=256):
    """Build the TensorRT engine; precisions are the ONNX graph's own (strongly typed)."""
    import tensorrt as trt

    logger = _trt_logger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 3, side, side), (4, 3, side, side), (MAX_CROPS, 3, side, side))
    profile.set_shape("intrinsics", (1, 3, 3), (4, 3, 3), (MAX_CROPS, 3, 3))
    n = len(vertices)
    profile.set_shape("landmarks", (1,), (min(n, 35),), (n,))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT could not build the NLF engine")
    write_engine(engine_path, serialized,
                 {"vertices": [int(v) for v in vertices], "precision": "fp16" if half else "fp32",
                  "max_crops": MAX_CROPS, "tensorrt": trt.__version__})


def export_engine(nlf_path, cano_path, vertices, engine_path, half=True, device="cuda:0"):
    """TorchScript release to TensorRT engine for ``vertices``, via ONNX beside the engine."""
    import numpy as np
    release = load_release(nlf_path)
    cano = torch.from_numpy(np.load(cano_path)[list(vertices)]).float()
    onnx_path = Path(engine_path).with_suffix(".onnx")
    export_onnx(release, cano, onnx_path, half, device)
    build_engine(onnx_path, engine_path, vertices, half)
    return onnx_path


class CropEngine:
    """The TensorRT engine, run on torch CUDA tensors without leaving the GPU."""

    def __init__(self, engine_path, device="cuda:0"):
        import tensorrt as trt
        data = Path(engine_path).read_bytes()
        meta = checked_meta(engine_path, data, trt.__version__)
        self.vertices, self.max_crops = meta["vertices"], meta["max_crops"]
        self.precision = meta["precision"]
        self.device = torch.device(device)
        self._engine = trt.Runtime(_trt_logger()).deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not load the NLF engine {engine_path}; {REBUILD}")
        self._context = self._engine.create_execution_context()

    def __call__(self, image, intrinsics, landmarks):
        """(B, 3, 256, 256) crops in [0, 1], (B, 3, 3), (P,) rows -> (B, P, 3) mm, (B, P)."""
        b, p = image.shape[0], landmarks.shape[0]
        if b > self.max_crops:
            raise ValueError(f"{b} crops in one pass, the NLF engine takes at most {self.max_crops}")
        ctx = self._context
        inputs = {"image": image.to(self.device, torch.float32).contiguous(),
                  "intrinsics": intrinsics.to(self.device, torch.float32).contiguous(),
                  "landmarks": landmarks.to(self.device, torch.int64).contiguous()}
        for name, tensor in inputs.items():
            if not ctx.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"the NLF engine does not take {name} of shape {tuple(tensor.shape)}")
            ctx.set_tensor_address(name, tensor.data_ptr())
        points = torch.empty((b, p, 3), device=self.device, dtype=torch.float32)
        uncertainty = torch.empty((b, p), device=self.device, dtype=torch.float32)
        ctx.set_tensor_address("points_mm", points.data_ptr())
        ctx.set_tensor_address("uncertainty", uncertainty.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("NLF TensorRT engine failed to run")
        stream.synchronize()
        return points, uncertainty
