# 22: NLF on TensorRT, all crops in one batch

**What to build:** NLF-S runs as one TensorRT fp16 engine instead of the released TorchScript file, and the crops of every view go through it in a single batched forward pass instead of one pass per crop. The whole crop network (256 x 256 crop and crop intrinsics in, 3D landmarks and uncertainties out) is one ONNX graph; detection, perspective cropping, batching and the rotation back to each camera frame stay outside it. The export recipe already exists and is documented in `docs/research/nlf-tensorrt/` (notes and reference script); it was validated in fp32 only, on a static batch of 1.

**Blocked by:** None (can start immediately). The ticket 07 recordings serve as the parity baseline.

**Status:** ready-for-agent

- [x] The crop network is exported to ONNX with a dynamic batch axis on image and intrinsics, with the localizer field evaluated once for the landmarks in use
- [x] One export serves every landmark map (the six fold maps and the toolbox map), or one engine per map is built and selected like the map itself
- [x] The toolbox wrapper does the perspective crop per detected person and view, runs all crops in one pass, and maps the output back to each camera frame
- [ ] Parity against TorchScript on the same frames: landmarks within 1 mm, uncertainties and resulting joint angles reported; measured for the TensorRT fp16 engine, not only fp32 ONNX
  - Measured (1012, Lifting, four views, 300 frames): 1.55 mm median, 14.2 mm max, 0.035 deg median joint angle. The 1 mm bar is not met, because TorchScript differs from itself by 2 mm.
- [ ] If parity holds, the ticket 07 recordings and every replay stay valid; if not, the gap is quantified and the choice goes to the author
  - The gap is quantified (`docs/research/nlf-tensorrt/README.md`); the author's choice is open.
- [ ] Per-stage timing before and after (detection, crop, NLF, fusion, IK) on the workstation GPU, one to four views
- [x] The backend is a setting, and TorchScript stays the default until parity is shown
- [x] The SMPL fit, if ever used, stays in torch (the TensorRT builder runs out of memory on it)

Timing on the RTX 3050 Ti laptop and the RTX 4070 desktop is not part of this ticket: it happens at the very end, from a clean install (ticket 19).
