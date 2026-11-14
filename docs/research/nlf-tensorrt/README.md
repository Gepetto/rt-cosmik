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

## Ticket 22: what was built and measured

Code: `src/rtcosmik/nlf/crop_network.py` (export, engine build, engine runtime), `src/rtcosmik/nlf/crops.py` (perspective crop and the rotation back), `backend="tensorrt"` in `NLFEstimator`, selected by `settings.nlf_backend`. Build the engine with `scripts/python/core/export_nlf_engine.py`.

- **One engine for every landmark map.** The engine holds the localizer weights of the union of the vertices of the toolbox map, the submitted map and the six fold maps (77 vertices). A third input, `landmarks`, selects the rows of the map in use right after the localizer convolution. The selection has to happen inside the graph: the absolute reconstruction is a least-squares fit over the queried landmarks, so querying the union and discarding rows afterwards would change every landmark.
- **Precision.** The backbone and head layer run in fp16; the localizer convolution, the soft-argmax and the reconstruction stay in fp32 (strongly typed network, the casts are in the ONNX graph). Millimetre coordinates at 3 m do not fit fp16.
- **Dynamic axes.** Batch 1 to 8 crops, 1 to 77 landmarks. `onnxsim` is not used.
- **The wrapper is exact.** Crops made by `crops.py`, sent through the TorchScript crop model and rotated back, match the full TorchScript call to 0.0003 mm. `crops.py` restates what the release does around its crop network for a single pass: no test-time augmentation, no lens distortion.
- **Export substitutions.** `BatchRenorm2d` is `BatchNorm2d` at inference; `linalg.inv` has no ONNX op and `nan_to_num` exports through float64, hence the closed-form 3x3 inverse and the clamped masked mean. Loading the release needs `torchvision` imported, for `torchvision::nms`.
- **Engine and sidecar.** `<engine>.json` records the vertices, the precision, the maximum number of crops, the TensorRT version and the sha256 of the engine file. Loading refuses an engine whose sidecar is missing, belongs to another file or names another TensorRT version, a file TensorRT cannot deserialize, and more crops than the engine was built for; each error says to rebuild with `scripts/python/core/export_nlf_engine.py`.
- **Provenance.** `run_info.json` has an `estimator` entry: NLF backend, detection mode, and for TensorRT the engine's file name, sha256, sidecar and the installed TensorRT version; for tiled detection the image size and the detector engine. Views recordings carry the same stamp, and a replay records the recording's stamp, not the replay-time settings; a recording without it is TorchScript with per-view detection.
- **Parity reference.** `nlf_backend_parity.py` always compares against TorchScript with per-view detection, whatever `settings.py` selects. The TorchScript file without `optimize_for_inference` is compared too, to show how far the baseline is from itself.

Parity on participant 1012, Lifting, cameras 0-2-4-6, first 300 frames, 35 landmarks, same boxes (`scripts/python/paper/nlf_backend_parity.py`). Differences to the TorchScript estimator as the toolbox runs it (fp16, `optimize_for_inference`, TF32 on):

| Compared with TorchScript | Landmark median | p99 | max | Joint angle median | max |
|---|---|---|---|---|---|
| TensorRT fp16 engine | 1.55 mm | 5.9 mm | 14.2 mm | 0.035 deg | 0.64 deg |
| TensorRT fp32 engine | 2.21 mm | 8.5 mm | 15.5 mm | 0.042 deg | 1.18 deg |
| TorchScript without `optimize_for_inference` | 2.17 mm | 9.1 mm | 15.0 mm | 0.052 deg | 1.14 deg |

Uncertainties: median difference 7e-5 (0.24 %), max 0.0026, for both engines.

**The 1 mm bar is not met**, and no engine can meet it against this baseline: the TorchScript model differs from itself by 2 mm (median) once `optimize_for_inference` is left out. The difference is almost all depth along the crop camera's axis, which NLF solves by least squares from the heatmaps and which amplifies rounding. On the same crops, against the fp32 crop model without TF32 (40 frames): fp32 engine 0.22 mm median, 0.62 mm max; TorchScript fp16 0.58 mm, 4.7 mm; fp16 engine 0.77 mm, 6.7 mm.
