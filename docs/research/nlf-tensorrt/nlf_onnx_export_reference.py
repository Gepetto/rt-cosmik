#!/usr/bin/env python3
"""NLF (Sárándi & Pons-Moll, NeurIPS'24) as a Competitor: its crop network, exported to ONNX.

Reference copy from the competitors_benchmark repository (competitors/nlf.py), kept for
ticket 22. It is not part of the toolbox and its imports resolve only in that repository.

    conda activate sam3d_video_312
    python competitors/nlf.py --size s --points direct   # all 6,890 SMPL vertices, no fit
    python competitors/nlf.py --size s --points subset   # NLF's own 1,024 vertices + 24 joints

Two variants are timed (ADR 0001, amendment): `direct` predicts every Posed mesh vertex from the
localizer field; `subset` is the network half of NLF's released recipe (1,024 points, then an SMPL
fit, then the SMPL forward, which are timed as post-processing).

The network is NLF's own crop model, taken from the released multi-person TorchScript file
(backbone, head layer and localizer field weights, in fp32). The localizer field is evaluated once,
outside the graph, for the chosen canonical points (NLF paper, Appendix D). Everything after the
backbone is restated in plain PyTorch so that it exports; the two steps ONNX has no op for,
`linalg.inv` and the Cholesky solve of 3x3 systems, are replaced by closed-form 3x3 solutions (the
same maths). Parity is measured against the release's own `predict_multi_same_canonicals`.

Input: a 256x256 RGB crop in [0, 1] (NLF normalises inside its backbone) and the crop camera's
intrinsics. Output: the points in camera space (millimetres) and their uncertainties.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "third_party" / "weights" / "nlf"
MODELS = ROOT / "models"
RELEASES = {"s": WEIGHTS / "nlf_s_multi_0.2.2.torchscript",
            "l": WEIGHTS / "nlf_l_multi_0.3.2.torchscript"}
SMPL_VERTICES = 6890
PARITY_BAR = 1e-4   # relative to the output's scale, see parity()


def eager_backbone(release) -> torch.nn.Module:
    """NLF's EfficientNetV2 (its own port: TensorFlow-style fixed padding, so not torchvision's),
    as a plain module with the release's weights. NLF's BatchRenorm2d is BatchNorm at inference.
    Its two training-stack imports are shimmed; the result matches the scripted backbone exactly."""
    import importlib.util
    import types
    from functools import partial

    florch = types.ModuleType("florch")
    florch.layers = types.ModuleType("florch.layers")
    florch.layers.BatchRenorm2d = torch.nn.BatchNorm2d
    spu = types.ModuleType("simplepyutils")
    spu.FLAGS = types.SimpleNamespace(centered_stride=bool(release.crop_model.heatmap_head.centered_stride))
    saved = {k: sys.modules.get(k) for k in ("florch", "florch.layers", "simplepyutils")}
    sys.modules.update({"florch": florch, "florch.layers": florch.layers, "simplepyutils": spu})
    try:
        spec = importlib.util.spec_from_file_location(
            "nlf_efficientnet", ROOT / "third_party" / "nlf" / "nlf" / "pt" / "backbones" / "efficientnet.py")
        eff = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eff)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    size = "s" if sum(p.numel() for p in release.crop_model.backbone.parameters()) < 50e6 else "l"
    features = getattr(eff, f"efficientnet_v2_{size}")(
        norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3), weights=None).features
    sd = release.crop_model.backbone.state_dict()
    features.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("1.")}, strict=True)
    return features.eval()


class TFPreproc(torch.nn.Module):
    """NLF's `tf_preproc`: [0, 1] -> [-1, 1]."""

    def forward(self, x):
        return 2 * x - 1


def eager_head_layer(release) -> torch.nn.Module:
    """Conv 1x1 (1280 -> 512, no bias), BatchRenorm (= BatchNorm at inference, eps 1e-3), SiLU."""
    sd = release.crop_model.heatmap_head.layer.state_dict()
    conv = sd["0.weight"]
    layer = torch.nn.Sequential(torch.nn.Conv2d(conv.shape[1], conv.shape[0], 1, bias=False),
                                torch.nn.BatchNorm2d(conv.shape[0], eps=1e-3), torch.nn.SiLU())
    layer.load_state_dict(sd, strict=True)
    return layer.eval()


def load_release(size: str) -> torch.jit.ScriptModule:
    import torchvision  # noqa: F401  registers torchvision::nms, which the file references

    m = torch.jit.load(str(RELEASES[size]), map_location="cpu")
    return m.float().eval()


def nlf_canonical_smpl_vertices() -> np.ndarray:
    """NLF's canonical SMPL vertices, regenerated as NLF does (`canonical_space.save_canonical_verts`):
    SMPL-neutral in NLF's canonical pose (hips +-40 deg, collarbones and shoulders +-15 deg about z),
    then averaged with its left-right mirror image under an optimal vertex matching."""
    import scipy.optimize
    sys.path.insert(0, str(ROOT))
    from competitors.body_models import load_smpl, smpl_vertices

    pose = np.zeros(72)
    pose[[5, 8]] = np.deg2rad([40, -40])            # lhip, rhip (z)
    pose[[41, 44]] = np.deg2rad([15, -15])          # lcla, rcla (z)
    pose[[50, 53]] = np.deg2rad([15, -15])          # lsho, rsho (z)
    verts = smpl_vertices(load_smpl(), pose)
    mirror = verts * np.array([-1.0, 1.0, 1.0])
    dist = np.linalg.norm(mirror[None] - verts[:, None], axis=-1)
    _, mirror_idx = scipy.optimize.linear_sum_assignment(dist)
    return ((verts + mirror[mirror_idx]) / 2).astype(np.float32)


def canonical_points(release, points: str, smpl_canonical: Path | None) -> tuple[torch.Tensor, str]:
    """The canonical query points: NLF's 1,024 SMPL vertices + 24 joints, or all 6,890 vertices."""
    if points == "subset":
        return release.cano_all["smpl"].float(), "NLF release: 1,024-vertex subset + 24 joints"
    if smpl_canonical is not None:
        return torch.from_numpy(np.load(smpl_canonical)).float(), f"{smpl_canonical.name}"
    cache = MODELS / "nlf_canonical_smpl_vertices.npy"
    if not cache.exists():
        MODELS.mkdir(exist_ok=True)
        np.save(cache, nlf_canonical_smpl_vertices())
    return torch.from_numpy(np.load(cache)).float(), "regenerated from SMPL-neutral (NLF's recipe)"
    # PLACEHOLDER until SMPL-neutral is available: the point coordinates do not change the graph
    # or its cost, only which surface points are predicted. Never reported.
    g = torch.Generator().manual_seed(0)
    cano = release.cano_all["smpl"][:1024].float()
    idx = torch.randint(0, len(cano), (SMPL_VERTICES,), generator=g)
    return cano[idx] + 0.01 * torch.randn(SMPL_VERTICES, 3, generator=g), "PLACEHOLDER"


def inv3x3(m: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of (..., 3, 3) matrices (adjugate / determinant)."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    co = torch.stack((e * i - f * h, c * h - b * i, b * f - c * e,
                      f * g - d * i, a * i - c * g, c * d - a * f,
                      d * h - e * g, b * g - a * h, a * e - b * d), -1).unflatten(-1, (3, 3))
    det = a * co[..., 0, 0] + b * co[..., 1, 0] + c * co[..., 2, 0]
    return co / det[..., None, None]


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dims: List[int], keepdim: bool) -> torch.Tensor:
    v = valid.to(x.dtype)
    while v.dim() < x.dim():
        v = v.unsqueeze(-1)
    # = nan_to_num(sum / count): with no valid item the sum is 0 too. nan_to_num exports via float64.
    return (x * v).sum(dims, keepdim=keepdim) / torch.clamp(v.sum(dims, keepdim=keepdim), min=1.0)


class NLFCrop(torch.nn.Module):
    """NLF's crop prediction for a fixed set of canonical points, in exportable form."""

    def __init__(self, release, cano: torch.Tensor):
        super().__init__()
        crop = release.crop_model
        head = crop.heatmap_head
        self.backbone = torch.nn.Sequential(TFPreproc(), eager_backbone(release))
        self.layer = eager_head_layer(release)
        self.proc_side, self.stride = int(head.proc_side), int(head.stride_test)
        self.centered = bool(head.centered_stride)
        self.box_size_m, self.depth = float(head.box_size_m), int(head.depth)
        self.uncert_bias, self.uncert_bias2 = float(head.uncert_bias), float(head.uncert_bias2)
        self.uncert_factor = 1.0 if bool(head.fix_uncert_factor) else 3.0
        with torch.no_grad():   # the localizer field, evaluated once for these points
            w = head.weight_field(cano)
            w = w.unflatten(-1, (int(head.backbone_link_dim) + 1, 2 + self.depth))
        self.register_buffer("w", w[:, :-1, :].permute(0, 2, 1).flatten(0, 1)[..., None, None].contiguous())
        self.register_buffer("b", w[:, -1, :].reshape(-1).contiguous())

    def _to_image(self, c: torch.Tensor) -> torch.Tensor:
        last = self.proc_side - 1
        out = c * (last - last % self.stride)
        if self.centered:
            out = out + self.stride // 2
        return out

    def _decode(self, heatmap: torch.Tensor, dims: List[int]) -> torch.Tensor:
        """Expected coordinate along each of `dims` (in [0, 1]), in the order given."""
        out: List[torch.Tensor] = []
        for d in dims:
            other: List[int] = []
            for x in dims:
                if x != d:
                    other.append(x)
            marginal = heatmap.sum(other, keepdim=True)
            n = heatmap.shape[d]
            coords = torch.arange(n, dtype=torch.float32, device=heatmap.device) / (n - 1)   # linspace(0, 1, n); exports as float
            shape: List[int] = []
            for i in range(heatmap.dim()):
                shape.append(n if i == d else 1)
            out.append((marginal * coords.reshape(shape)).sum(d, keepdim=True).flatten(2))
        return torch.cat(out, -1)

    def _in_fov(self, coords: torch.Tensor, border: float) -> torch.Tensor:
        offset = 0.0
        if not self.centered:
            offset = -self.stride / 2.0
        lo, hi = self.stride * border + offset, self.proc_side - self.stride * border + offset
        return ((coords >= lo) & (coords <= hi)).all(-1)

    def forward(self, image: torch.Tensor, intrinsics: torch.Tensor):
        feats = self.layer(self.backbone(image))
        logits = F.conv2d(feats, self.w, self.b).unflatten(1, (-1, 2 + self.depth))  # N P C H W
        n, p, h, wd = logits.shape[0], logits.shape[1], logits.shape[3], logits.shape[4]
        unc_map = logits[:, :, 0]
        xy_metric = self._decode(torch.softmax(logits[:, :, 1].flatten(2), -1).reshape(n, p, h, wd), [3, 2])
        hm25 = torch.softmax(logits[:, :, 2:].flatten(2), -1).reshape(n, p, self.depth, h, wd)
        coords25 = self._decode(hm25, [4, 3, 2])
        unc = F.softplus((unc_map * hm25.sum(2)).sum((2, 3)) + self.uncert_bias) + self.uncert_bias2

        coords2d = self._to_image(coords25[..., :2])
        rel = torch.cat((self._to_image(xy_metric) * self.box_size_m / self.proc_side,
                         coords25[..., 2:] * self.box_size_m), -1)
        return self._absolute(coords2d, rel, intrinsics, unc < 0.3) * 1000, unc * self.uncert_factor

    def _absolute(self, coords2d: torch.Tensor, rel: torch.Tensor, k: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
        """`ptu3d.reconstruct_absolute` (full perspective, mix 0.5, border factors 1.0 / 0.6)."""
        ones = torch.ones_like(coords2d[..., :1])
        norm2d = (torch.cat((coords2d, ones), -1) @ inv3x3(k).transpose(1, 2))[..., :2]
        valid1 = self._in_fov(coords2d, 1.0) & valid

        # reconstruct_ref_fullpersp, with the 3x3 normal equations solved in closed form
        n_pts = norm2d.shape[1]
        s2 = torch.sqrt(masked_mean(norm2d.square(), valid1, [1, 2], True) + 1e-10)
        a2d = (norm2d / s2).reshape(-1, n_pts * 2, 1)
        eyes = torch.eye(2, dtype=norm2d.dtype, device=norm2d.device).repeat(n_pts, 1).unsqueeze(0)
        A = torch.cat((eyes, -a2d), 2)
        back = norm2d * rel[:, :, 2:] - rel[:, :, :2]
        sb = torch.sqrt(masked_mean(back.square(), valid1, [1, 2], True) + 1e-10)
        bvec = (back / sb).reshape(-1, n_pts * 2, 1)
        wts = torch.repeat_interleave(valid1.to(norm2d.dtype) + 1e-8, 2, 1)
        wA = wts.unsqueeze(-1) * A
        gram = wA.transpose(1, 2) @ A + 1e-4 * torch.eye(3, dtype=A.dtype, device=A.device)
        ref = inv3x3(gram) @ (wA.transpose(1, 2) @ bvec)
        sb, s2 = sb.reshape(-1, 1, 1), s2.reshape(-1, 1, 1)
        ref = torch.cat((ref[:, :2] * sb, ref[:, 2:] * (sb / s2)), 1).squeeze(-1)

        abs3d = rel + ref.unsqueeze(1)
        abs2d = torch.cat((norm2d, ones), -1) * (rel[..., 2] + ref[:, 2:3]).unsqueeze(-1)
        abs2d = 0.5 * abs3d + 0.5 * abs2d
        proj = abs3d / torch.clamp(abs3d[..., 2:], min=0.1)
        proj = torch.einsum("bnk,bjk->bnj", proj, k[..., :2, :])
        valid2 = self._in_fov(proj, 0.6) & (abs3d[..., 2] > 0.001)
        return torch.where(valid2.unsqueeze(-1), abs2d, abs3d)


def crop_intrinsics(fov_deg: float = 55.0, side: int = 256) -> torch.Tensor:
    f = side / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    c = (side - 1) / 2
    return torch.tensor([[[f, 0, c], [0, f, c], [0, 0, 1]]], dtype=torch.float32)


def parity_inputs(side: int) -> torch.Tensor:
    """The 40 parity crops (same images as g8h's), un-normalised to [0, 1] and resized to NLF's side."""
    sys.path.insert(0, str(ROOT))
    from competitors.parity_crops import unit_rgb
    rgb = torch.from_numpy(unit_rgb())
    return F.interpolate(rgb, size=(side, side), mode="bilinear", align_corners=False)


def parity(release, model, onnx_path, cano, images, k) -> dict:
    """Max-abs-diff of the ONNX file (ORT CPU, no graph optimisation) and the plain-PyTorch
    restatement against the release's own TorchScript prediction, over all crops."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    worst = {"restated_points_mm": 0.0, "onnx_points_mm": 0.0, "onnx_uncert": 0.0}
    with torch.no_grad():
        for img in images:
            x = img[None]
            ref_pts, ref_unc = release.crop_model.predict_multi_same_canonicals(x, k, cano)
            mine_pts, _ = model(x, k)
            got_pts, got_unc = sess.run(None, {"image": x.numpy(), "intrinsics": k.numpy()})
            worst["restated_points_mm"] = max(worst["restated_points_mm"], float((mine_pts - ref_pts).abs().max()))
            worst["onnx_points_mm"] = max(worst["onnx_points_mm"], float(np.abs(got_pts - ref_pts.numpy()).max()))
            worst["onnx_uncert"] = max(worst["onnx_uncert"], float(np.abs(got_unc - ref_unc.numpy()).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=["s", "l"], default="s")
    ap.add_argument("--points", choices=["direct", "subset"], default="direct")
    ap.add_argument("--smpl-canonical", type=Path, help="(6890, 3) NLF canonical SMPL vertices")
    args = ap.parse_args()

    release = load_release(args.size)
    cano, cano_source = canonical_points(release, args.points, args.smpl_canonical)
    model = NLFCrop(release, cano).eval()
    side = model.proc_side
    k = crop_intrinsics(side=side)

    model_id = f"nlf-{args.size}-{args.points}"
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"{model_id}.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 3, side, side), k), str(path),
                          input_names=["image", "intrinsics"], output_names=["points_mm", "uncertainty"],
                          opset_version=17, dynamo=False)
    import onnx
    import onnxsim
    simplified, ok = onnxsim.simplify(onnx.load(str(path)))
    if ok:
        onnx.save(simplified, str(path))

    images = parity_inputs(side)
    diffs = parity(release, model, path, cano, images, k)
    # NLF's points are in millimetres at ~3-5 m depth; the bar is 1e-4 relative, i.e. 0.5 mm.
    bar_mm = 0.5
    report = {"model": model_id, "points": int(cano.shape[0]), "canonical_points": cano_source,
              "input_side": side, "parity_max_abs": diffs, "bar_mm": bar_mm, "n_crops": len(images),
              "simplified": bool(ok), "file_mb": round(path.stat().st_size / 1e6, 1)}
    path.with_suffix(".parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    worst = max(diffs["restated_points_mm"], diffs["onnx_points_mm"])
    print(("[OK]" if worst < bar_mm else "[FAIL]") + f" worst {worst:.3e} mm vs bar {bar_mm} mm")
    return 0 if worst < bar_mm else 1


if __name__ == "__main__":
    sys.exit(main())
