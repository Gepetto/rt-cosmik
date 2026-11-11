# NLF-S on ONNX and TensorRT: what was already done

Notes for ticket 22. Source: the `competitors_benchmark` repository, `competitors/nlf.py` (class `NLFCrop`), copied here as `nlf_onnx_export_reference.py`. It ran there with `python competitors/nlf.py --size s --points subset`.

The backbone does not need to be split from the heads. The whole crop network, from the 256 x 256 crop to 3D points and uncertainties, exports as a single ONNX graph, and TensorRT takes all of it as one fp16 engine. Only the per-person orchestration (detection, cropping, batching) stays outside the graph.

## The export

- **Weights** come from the released TorchScript file: `torch.jit.load("nlf_s_multi_0.2.2.torchscript")`, then `.float().eval()`, taken from `release.crop_model`. Nothing is retrained.
- **Backbone**, rebuilt as an eager module from NLF's own EfficientNetV2 port (`nlf/pt/backbones/efficientnet.py` in the NLF repository). That port uses TensorFlow-style fixed padding, so torchvision's EfficientNetV2 does not match. Two training-stack imports are shimmed: `florch.layers.BatchRenorm2d` becomes `BatchNorm2d` (identical at inference), and `simplepyutils.FLAGS.centered_stride` is set. Weights load from `crop_model.backbone.state_dict()` (keys under `1.`) with `strict=True`. A `2x - 1` preprocessing step goes in front.
- **Head layer**, rebuilt the same way: Conv 1x1 (1280 to 512, no bias), BatchNorm (eps 1e-3), SiLU, loaded from `heatmap_head.layer`.
- **Localizer field**, evaluated once, outside the graph, and stored as a constant: `head.weight_field(cano)` on the fixed canonical points. Its output becomes the weights and bias of a plain `conv2d` on the features (NLF paper, App. D). This is the key step: the field MLP no longer runs per frame, and the output size is fixed. `cano` can be any landmark set.
- **Everything after the backbone** is restated in plain PyTorch: soft-argmax decoding (2D and 2.5D heatmaps), the uncertainty softplus, and `reconstruct_absolute` (full perspective, mix 0.5, FOV border factors 1.0 / 0.6). Two steps have no ONNX op and are replaced by the same maths in closed form: `torch.linalg.inv` becomes a 3x3 adjugate/determinant inverse, and the Cholesky solve of the 3x3 normal equations becomes that same inverse. `nan_to_num` is replaced by a clamped masked mean, because it exports through float64.
- **Export call**: `torch.onnx.export`, opset 17, `dynamo=False`, static batch 1, then `onnxsim`.
- **Inputs**: `image` (1, 3, 256, 256) in [0, 1], and `intrinsics` (1, 3, 3) of the crop camera.
- **Outputs**: `points_mm` (crop-camera space, millimetres) and `uncertainty`.

## Parity measured there

Against the release's own `crop_model.predict_multi_same_canonicals(x, K, cano)`, on 40 crops:

| Variant | Worst difference |
|---|---|
| Eager restatement | 0.0013 mm |
| ONNX (ONNX Runtime CPU, fp32, graph optimisation off) | 0.014 mm |

## TensorRT

Run through ONNX Runtime's TensorRT execution provider, TensorRT 10.15: `trt_fp16_enable=True`, `trt_builder_optimization_level=3`, engine and timing caches on. The whole graph became one engine in 1.7 s.

On an RTX 4070, batch 1, all 6,890 vertices:

| Backend | Latency |
|---|---|
| TensorRT fp16 | 1.52 ms |
| ONNX Runtime CUDA fp32 | 3.71 ms |
| Original PyTorch | 10.19 ms |

## What is left for ticket 22

- **fp16 accuracy was never checked.** Parity was measured only for fp32 ONNX on CPU. Run the ONNX file through the TensorRT provider and compare with TorchScript on the same crops. The fp32 restatement matches to about 0.01 mm, so any visible gap comes from fp16.
- **The multi-person wrapper is not in the graph**: YOLO boxes; NLF's perspective crop (warping to a rotated virtual camera, 256 px, and its intrinsics); batching people across the cameras; rotating the output points from the crop camera back to the camera frame.
- **Batching**: the export is static batch 1. For several crops per pass, re-export with a dynamic batch axis on `image` and `intrinsics`, or with a fixed maximum batch. The graph handles batches already; that export is untested.
- **The SMPL fit cannot go on TensorRT**: the builder ran out of host memory on it (about 30 GB). If the fit is ever used, it stays in torch.
- **Old recordings are only safe if the outputs match.** Check the landmarks against TorchScript before relying on existing recordings.
