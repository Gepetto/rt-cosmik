# Installation

RT-COSMIK runs on Linux with an NVIDIA GPU. The simplest way to get its full
software stack is the Docker image: one script builds it and starts it. VS Code
can open the same image as a dev container, and the image's recipe doubles as
the reference for a native install.

## Requirements

- Linux on x86-64. Ubuntu 22.04 is the tested system.
- An NVIDIA GPU and its driver. The image uses CUDA 12.1.
- For Docker: [Docker Engine](https://docs.docker.com/engine/install/) and, for
  GPU runs, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- For live capture: USB webcams that stream MJPEG (see the
  [README](own-cameras.md)).

## Docker

```bash
git clone https://github.com/Gepetto/rt-cosmik.git
cd rt-cosmik
docker/run.sh
```

The first call builds the image `rt-cosmik:latest` from
[`docker/Dockerfile`](https://github.com/Gepetto/rt-cosmik/blob/main/docker/Dockerfile). It compiles CasADi, Pinocchio and
acados from source, so it can take around an hour on a workstation and needs
memory: compile jobs are capped by the RAM available, and if the build is still
killed for lack of memory, add swap.

Each call then starts a container with:

- the GPU;
- every device under `/dev`, so the cameras;
- the host network, so the 3D viewer's URL opens in the host's browser;
- the host's display, for the OpenCV windows of camera calibration and preview;
- your checkout mounted at `/root/workspace/rt-cosmik`.

The container is removed when you exit it. Everything worth keeping lives in
the checkout: models in `weights/`, solvers in `ocp/`, calibrations in
`config/cam_params/`, results in `output/`, the sample in `data/`. The container
runs as root, but what it writes there is handed back to you when it exits, so
you can edit or delete it without sudo.

Other uses:

```bash
docker/run.sh python3 -m pytest tests/unit   # run one command in the container
docker/run.sh --build                        # rebuild the image, e.g. after a Dockerfile change
docker/run.sh --cpu python3 -m pytest tests/unit  # without NVIDIA devices or drivers
```

**Companion repositories.** Clone them next to `rt-cosmik`, and `docker/run.sh`
mounts them as well:

```
<your folder>/
├── rt-cosmik/
├── cams_calibration/     ->  /root/workspace/cams_calibration
└── rtcosmik_ros/         ->  /root/workspace/ros_ws/src/rtcosmik_ros
```

The same image supports the standalone toolbox and the
[ROS overlay](https://github.com/Gepetto/rtcosmik_ros), including RViz,
`robot_state_publisher`, TF and the required messages. From the host:

```bash
git clone https://github.com/Gepetto/rtcosmik_ros.git ../rtcosmik_ros
docker/run.sh
```

Then, in the container:

```bash
cd /root/workspace/ros_ws
colcon build --packages-select rtcosmik_ros
source install/setup.bash
ros2 launch rtcosmik_ros start.launch.py
```

Set `RTCOSMIK_ROS_DIR=/absolute/path/to/rtcosmik_ros` when the overlay is
elsewhere. The overlay remains optional for standalone use. Its build and
install directories are local to the container; rebuild it in each new session.

### What the image contains

The pinned versions preserve the working environment used during development;
they are not minimum requirements or a claim that newer versions are
incompatible. Source versions are build arguments in `docker/Dockerfile`, and
Python package pins are in `docker/requirements.txt`. When upgrading, run
`docker/check.sh` to check imports, solver generation, the sample pipeline and
the ROS overlay. TensorRT changes also require regenerating detector engines.

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| Python, numpy | 3.10, 1.26.4 |
| PyTorch | 2.4.1 (CUDA 12.1) |
| TensorRT | 10.15.1.29 |
| ultralytics (person detector) | 8.4.14 |
| ROS 2 | Humble (ROS base, RViz and overlay build tools) |
| CasADi | 3.7.2, with fatrop and IPOPT |
| fatrop, blasfeo | commits `45ee388`, `9923ac8` |
| eigenpy, HPP-FCL | ROS Humble binary packages (exact versions in `/opt/manifest/dpkg.txt`) |
| Pinocchio | 3.9.0, with CasADi and collision support |
| example-robot-data | commit `9c202b1` (provides the human model) |
| acados | commit `8e1a6f8` (v0.5.4-20) |

Eigenpy and HPP-FCL come from the ROS Humble APT repository. Pinocchio 3.9
requires the older HPP-FCL interface, which the ROS coal package does not
expose. Pinocchio is compiled against those libraries with CasADi support,
which the standard ROS package does not provide. The custom CasADi/fatrop and
acados builds are kept for the solver backends, together with the pinned
example-robot-data human model.

Python dependencies are installed with a pinned `uv` in a virtual environment at `/usr/local`,
alongside the source-built Python bindings. It can read the system packages
needed by ROS, and a `dist-packages` alias makes its packages visible to ROS
scripts that use `/usr/bin/python3`. The mounted checkout is registered with a
`.pth` file, as are ROS's Python package directories; no shell activation or
extra `PYTHONPATH` is needed for these imports. Native
libraries are registered with `ldconfig`.

APT and Python downloads use BuildKit cache mounts. C/C++ compilation uses
`ccache` with a separate cache mount. Application requirements and `uv` are
installed after the native libraries: updating them does not rebuild CasADi or
Pinocchio. PyTorch and TensorRT also precede the application requirements, so
utility updates reuse their large layers. NumPy is installed before compilation
because its ABI matters to the bindings. The CUDA runtime base omits the CUDA
compiler, profiler and development
archives; C/C++ compilers and acados headers remain available for solver generation.
ONNX Runtime and torchaudio are retained for development.

Source downloads use Docker
`ADD`, with release tarballs or pinned commit archives and the required
submodules. Temporary source and build trees are removed after installation;
acados sources remain available for runtime solver generation. The final
environment and source archive checksums are recorded in `/opt/manifest`.

To override the automatic compile-job limit when building directly:

```bash
docker build --build-arg MAKE_JOBS=2 -t rt-cosmik:latest docker
```

## VS Code dev container

With the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
extension, open the checkout and run **Dev Containers: Reopen in Container**. It
builds the same image and opens the checkout inside it, with the same access to
the GPU, the cameras, the network and the display
([`.devcontainer/devcontainer.json`](https://github.com/Gepetto/rt-cosmik/blob/main/.devcontainer/devcontainer.json)). The
companion repositories are not mounted automatically; add them to `runArgs` if
you need them. Files written from the dev container stay root's on the host
until `docker/run.sh true`, run from the host, hands them back.

## Native install

Without Docker, follow [`docker/Dockerfile`](https://github.com/Gepetto/rt-cosmik/blob/main/docker/Dockerfile) on Ubuntu
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
docker/check.sh --cpu                                   # builds and checks without CUDA
docker/check.sh --gpu                                   # also runs videos and the ROS replay
```

With no mode, `docker/check.sh` selects GPU mode when `nvidia-smi` detects a
GPU, otherwise CPU mode. `--gpu` requires working GPU access in the container;
it does not fall back to CPU. Use `--no-build` to check an existing image, and
`RTCOSMIK_IMAGE` to select a tag other than `rt-cosmik:latest`.

Both modes check Python/native bindings, unit tests, solver generation and
manifests, and export a small synthetic Meshcat animation. When the overlay is
mounted, both also build it, import the bridge and inspect its launch arguments.
GPU mode additionally checks PyTorch, TensorRT and ONNX Runtime CUDA inference,
processes the sample videos, compares the results with motion capture, and
checks that the ROS replay publishes finite joint states with advancing timestamps.
The ROS check uses a private source copy and temporary settings, preserving
the checkout's scaled URDF and recording preferences.

Logs and `summary.txt` are written under `output/docker-check/`. The launcher
hands generated files back to the calling user, and the check verifies ownership.
GPU-dependent checks are explicitly marked `SKIP` in CPU mode; a missing overlay is also
reported as `SKIP`. Any failed check returns a nonzero exit status. A CPU pass
therefore validates the CPU checks and does not claim an end-to-end inference pass.
Viewing a Meshcat replay of saved results does not require CUDA.

Then run the [sample trial](getting-started.md).

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `No '<model>' detector engine for <N> camera(s)` | Build it, as the message says: `YOLO_MODELS=<model> BATCHES=<N> bash scripts/bash/fetch_models.sh`. |
| `The generated acados OCP in ... does not match the current configuration` | A setting the solver is built from changed (it lists which). Regenerate with `run_ocp_codegen.py`. |
| `Camera(s) [...] requested but not attached` | Check the cables and `v4l2-ctl --list-devices`, or request the cameras you have with `--cameras`. See [live.md](live.md#which-cameras-are-used). |
| The viewer page does not load | Use the URL the run prints: the port moves to 7001 and up when 7000 is taken. |
| OpenCV windows do not open from Docker | On the host, `echo $DISPLAY` must be set, and `xhost +si:localuser:root` lets the container draw. |
| `ImportError` on `pinocchio.casadi` | Pinocchio was built without CasADi support, or a pip CasADi wheel shadows the one it was built with. |
