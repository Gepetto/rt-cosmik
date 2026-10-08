# Installation

RT-COSMIK runs on Linux with an NVIDIA GPU. The simplest way to get its full
software stack is the Docker image: one script builds it and starts it. VS Code
can open the same image as a dev container, and the image's recipe doubles as
the reference for a native install.

## Requirements

- Linux on x86-64. Ubuntu 22.04 is the tested system.
- An NVIDIA GPU and its driver. The image uses CUDA 12.1.
- For Docker: [Docker Engine](https://docs.docker.com/engine/install/) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- For live capture: USB webcams that stream MJPEG (see the
  [README](../README.md#use-your-own-cameras)).

## Docker

```bash
git clone https://github.com/Gepetto/rt-cosmik.git
cd rt-cosmik
docker/run.sh
```

The first call builds the image `rt-cosmik:latest` from
[`docker/Dockerfile`](../docker/Dockerfile). It compiles CasADi, Pinocchio and
acados from source, so it takes a while and needs memory: compile jobs are capped
by the RAM available, and if the build is still killed for lack of memory, add
swap.

Each call then starts a container with:

- the GPU;
- every device under `/dev`, so the cameras;
- the host network, so the 3D viewer's URL opens in the host's browser;
- the host's display, for the OpenCV windows of camera calibration and preview;
- your checkout mounted at `/root/workspace/rt-cosmik`.

The container is removed when you exit it. Everything worth keeping lives in
the checkout: models in `weights/`, solvers in `ocp/`, calibrations in
`config/cam_params/`, results in `output/`, the sample in `data/`. These files
are created by root inside the container; `sudo chown -R $USER .` gives them
back to you on the host.

Other uses:

```bash
docker/run.sh python3 -m pytest tests/unit   # run one command in the container
docker/run.sh --build                        # rebuild the image, e.g. after a Dockerfile change
```

**Companion repositories.** Clone them next to `rt-cosmik`, and `docker/run.sh`
mounts them as well:

```
<your folder>/
├── rt-cosmik/
├── cams_calibration/     ->  /root/workspace/cams_calibration
└── rtcosmik_ros/         ->  /root/workspace/ros_ws/src/rtcosmik_ros
```

`rtcosmik_ros` is then built in the container with
`cd /root/workspace/ros_ws && colcon build --packages-select rtcosmik_ros`.

### What the image contains

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` |
| Python, numpy | 3.10, 1.26.4 |
| PyTorch | 2.4.1 (CUDA 12.1) |
| TensorRT | 10.15.1.29 |
| ROS 2 | Humble (desktop) |
| CasADi | 3.7.2, with fatrop and IPOPT |
| fatrop, blasfeo | commits `45ee388`, `9923ac8` |
| eigenpy, coal | 3.12.0, 3.0.2 |
| Pinocchio | 3.9.0, with CasADi and collision support |
| example-robot-data | commit `9c202b1` (provides the human model) |
| acados | commit `8e1a6f8` (v0.5.4-20) |

## VS Code dev container

With the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
extension, open the checkout and run **Dev Containers: Reopen in Container**. It
builds the same image and opens the checkout inside it, with the same access to
the GPU, the cameras, the network and the display
([`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)). The
companion repositories are not mounted automatically; add them to `runArgs` if
you need them.

## Native install

Without Docker, follow [`docker/Dockerfile`](../docker/Dockerfile) on Ubuntu
22.04: it is the tested recipe, step by step. The points that matter:

1. **numpy below 2**, which the compiled dependencies are built against.
2. **PyTorch with CUDA**, then `ultralytics`, `opencv-python`, `meshcat`,
   `pandas`, `quadprog`, `pynput`, `onnx` and `tensorrt`.
3. **Pinocchio with CasADi support** (`-DBUILD_WITH_CASADI_SUPPORT=ON`), and
   **example-robot-data**, which provides the human model.
4. **acados**, for the default solver backend. `scripts/bash/setup_env.sh --acados`
   builds it next to the repository and exports `ACADOS_SOURCE_DIR`. Install its
   Python interface with `--no-deps`: otherwise pip installs a CasADi wheel over
   the one Pinocchio was built with, and `pinocchio.casadi` breaks.
5. **ffmpeg** and **v4l-utils** (`sudo apt install ffmpeg v4l-utils`), which open
   and list the cameras.
6. **RT-COSMIK itself**: `pip install -e .` from the checkout, or
   `scripts/bash/setup_env.sh`, which makes `rtcosmik` importable for every
   interpreter, including ROS 2's, where the system setuptools is too old for
   editable installs.

## Models

```bash
scripts/bash/fetch_models.sh
```

This downloads the NLF and YOLO weights into `weights/`, and exports a TensorRT
engine of the detector chosen by `yolo_model` in `settings.py` for each camera
count from 1 to 6. Engines are built non-dynamic, so the batch size is fixed at
export time and the pipeline picks the engine matching the cameras in use. Build
a subset with `BATCHES="2 4"`, and engines of other detectors with
`YOLO_EXPORT_MODELS="yolo11n yolo26n"`.

An engine only works on the GPU model and with the TensorRT version that built
it. After changing either, delete `weights/yolo/*.engine` and run the script
again.

## Inverse kinematics solver

```bash
python3 scripts/python/core/run_ocp_codegen.py --backend acados --profile realtime
```

Only needed for the default moving-horizon inverse kinematics. One generated
solver serves every person; regenerate it after changing what it is built from.
See [inverse-kinematics.md](inverse-kinematics.md).

## Checking the installation

```bash
python3 -m pytest tests/unit -q                           # seconds; camera hardware tests are skipped
python3 scripts/python/core/run_ocp_codegen.py --check    # is the solver up to date?
```

Then run the [sample trial](../README.md#quick-start).

To check the whole Docker route at once, as a new user would follow it, run
`docker/check.sh` on the host. It builds the image, then, inside it, checks the
environment, fetches the models, generates the solver, runs the sample trial,
compares it with motion capture and runs the unit tests; with `rtcosmik_ros`
cloned next to the checkout, it also builds the ROS 2 node and replays the
sample through it. Each step is reported as PASS or FAIL in
`output/docker-check/<date>/summary.txt`, with its log next to it.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `No '<model>' detector engine for <N> camera(s)` | Build it, as the message says: `YOLO_MODELS=<model> BATCHES=<N> bash scripts/bash/fetch_models.sh`. |
| `The generated acados OCP in ... does not match the current configuration` | A setting the solver is built from changed (it lists which). Regenerate with `run_ocp_codegen.py`. |
| `Camera(s) [...] requested but not attached` | Check the cables and `v4l2-ctl --list-devices`, or request the cameras you have with `--cameras`. See [live.md](live.md#which-cameras-are-used). |
| The viewer page does not load | Use the URL the run prints: the port moves to 7001 and up when 7000 is taken. |
| OpenCV windows do not open from Docker | On the host, `echo $DISPLAY` must be set, and `xhost +si:localuser:root` lets the container draw. |
| `ImportError` on `pinocchio.casadi` | Pinocchio was built without CasADi support, or a pip CasADi wheel shadows the one it was built with. |
