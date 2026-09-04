# Fast SAM 3D Body pose estimator

RT-COSMIK's Fast SAM 3D Body adapter runs monocular inference using one
calibrated RGB camera. Its default configuration uses the fastest path provided
by the fork: a YOLO11m-Pose TensorRT detector, a DINOv3 TensorRT backbone, and
compiled body/hand decoders. Supplying the calibrated camera matrix removes the
MoGe/FOV network from the inference path.

The integration test uses:

- `tests/full/data/camera_0.mp4`
- `tests/full/config/c1_params_color.yaml`

## GPU container requirement

The environment needs an NVIDIA GPU exposed inside the running container. Check
this before building engines:

```bash
nvidia-smi
```

If that fails, restart the container with GPU passthrough (for example,
`docker run --gpus all ...`). TensorRT engines should be built on the same GPU
class and software stack used for inference.

## Install and build

The SAM 3D Body checkpoint is gated. Accept the license at
<https://huggingface.co/facebook/sam-3d-body-dinov3>, then run:

```bash
./scripts/bash/setup_fastsam3dbody.sh
.venv-fastsam3dbody/bin/huggingface-cli login
./scripts/bash/setup_fastsam3dbody.sh --download-models --build-tensorrt
```

The script installs a pinned, isolated environment at
`.venv-fastsam3dbody`, checks out the tested fork revision under
`../deps/Fast-SAM-3D-Body`, downloads the model files, and builds FP16
TensorRT engines for one tracked person. The DINOv3 profile covers the body and
two hand crops (dynamic backbone batches 1–3). It intentionally does not build
MoGe, because RT-COSMIK supplies camera intrinsics.

## Run the smoke test

```bash
MPLCONFIGDIR=/tmp/rtcosmik-matplotlib \
  .venv-fastsam3dbody/bin/python -m pytest -s \
  tests/unit/pose_estimator/test_fastsam3dbody.py
```

The GPU integration test collects six successful outputs from the first 20
frames, checks the MHR output contract (18,439 vertices, 70 keypoints, and 127
joints), and saves the final result under pytest's temporary directory. Without
a GPU or built engines, only the calibration and output-contract tests run and
the integration test is skipped.

On the RTX 4500 Ada used for this setup, the validated steady-state inference
times were 114–116 ms/frame. Model initialization takes about 15 seconds, and
the first two real frames take several seconds while `torch.compile` finishes
specializing the decoder and MHR graphs. A live pipeline should warm up with at
least two frames before publishing results. TF32 was tested and was slower for
this workload, so the adapter retains the fork's native FP32 decoder policy.

For a slower diagnostic fallback on a GPU without TensorRT engines, construct
`FastSAM3DBodyConfig(require_tensorrt=False)`. The default remains TensorRT.

## Export a complete video

The exporter defaults to `camera_0.mp4`, its `c1` calibration, and writes to
`output/fastsam3dbody/camera_0`:

```bash
MPLCONFIGDIR=/tmp/rtcosmik-matplotlib \
  .venv-fastsam3dbody/bin/python \
  scripts/python/export_fastsam3dbody_video.py
```

It saves one row per input frame in `bbox.npy`, `cam_t.npy`,
`focal_length.npy`, `joint_global_rots.npy`, `joints127_cam.npy`,
`keypoints70_2d.npy`, `keypoints70_cam.npy`, `reprojection_rmse_px.npy`, and
`vertices_cam.npy`. Static files contain mesh faces, MHR70 names/bones, and the
127-joint names/parents. `valid.npy` marks successful frames,
`metadata.json` records calibration and coordinate conventions, and
`keypoints2d_overlay.mp4` draws the 2D skeleton over the undistorted video.

Use `--start-frame`, `--end-frame`, or `--stride` for subsets. Existing output
is protected unless `--overwrite` is supplied.
