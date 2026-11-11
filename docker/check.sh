#!/usr/bin/env bash
# Build and check the image. CPU checks cover bindings, solvers, Meshcat and
# the optional ROS overlay; GPU checks also process videos and replay ROS.
#   docker/check.sh [--cpu | --gpu] [--no-build] [COPY_LOGS_TO]
# Without a mode, use GPU checks when nvidia-smi detects a GPU, otherwise CPU.
# RTCOSMIK_IMAGE selects the image; RTCOSMIK_ROS_DIR selects the ROS checkout.
set -euo pipefail

MODE=auto
BUILD=1
COPY_TO=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu|--gpu)
      [[ "${MODE}" == auto ]] || { echo 'Choose one of --cpu and --gpu.' >&2; exit 2; }
      MODE="${1#--}" ;;
    --no-build) BUILD=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) [[ -z "${COPY_TO}" ]] || { echo 'Only one log destination is accepted.' >&2; exit 2; }
       COPY_TO="$1" ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
LOGS_REL="output/docker-check/${RUN_ID}"
LOGS="${REPO_ROOT}/${LOGS_REL}"
mkdir -p "${LOGS}"

if [[ "${MODE}" == auto ]]; then
  MODE=cpu
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then MODE=gpu; fi
fi
RUN_ARGS=()
[[ "${MODE}" == cpu ]] && RUN_ARGS+=(--cpu)
printf 'Check mode: %s\nLogs: %s\n' "${MODE}" "${LOGS}"
{
  echo "RT-COSMIK Docker check ${RUN_ID}; mode=${MODE}"
  echo "commit: $(git -C "${REPO_ROOT}" log -1 --format='%h %s')"
  echo "host: $(uname -srm)"
  docker --version
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
} > "${LOGS}/host.txt"

cat > "${LOGS}/inside.sh" <<'INSIDE'
set -uo pipefail
cd /root/workspace/rt-cosmik
LOGS="$1"
MODE="$2"
FAILURES=0
export PYTHONDONTWRITEBYTECODE=1

step() {
  local name="$1" start=$SECONDS result
  shift
  echo "[..] ${name}"
  if "$@" > "${LOGS}/${name}.log" 2>&1; then
    printf 'PASS  %-12s %5ss\n' "${name}" "$((SECONDS - start))" | tee -a "${LOGS}/summary.txt"
  else
    result=$?
    printf 'FAIL  %-12s %5ss  exit %s; see %s.log\n' "${name}" "$((SECONDS - start))" "${result}" "${name}" \
      | tee -a "${LOGS}/summary.txt"
    FAILURES=1
    return "${result}"
  fi
}

skip() { printf 'SKIP  %-12s %s\n' "$1" "$2" | tee -a "${LOGS}/summary.txt"; }

environment() {
  ffmpeg -version > /dev/null && v4l2-ctl --version && python3 -m pip check || return
  /usr/bin/python3 -c 'import rclpy, rtcosmik, pinocchio.casadi, acados_template' || return
  python3 - "${MODE}" "${LOGS}/meshcat.html" <<'PY_ENV'
"""Exercise the image's native, Python and viewer stack, with optional CUDA."""

import sys
from pathlib import Path
from unittest.mock import patch

import casadi
import example_robot_data as erd
import meshcat
from meshcat.animation import Animation
from meshcat.visualizer import Visualizer
import numpy as np
import onnx
import onnxruntime as ort
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
import scipy.signal
import tensorrt as trt
import torch
import torchaudio

import acados_template  # noqa: F401
import rtcosmik  # noqa: F401


def check_native():
    assert np.__version__ == "1.26.4"
    print(f"NumPy {np.__version__}, CasADi {casadi.__version__}, Pinocchio {pin.__version__}")
    print(f"PyTorch {torch.__version__}, torchaudio {torchaudio.__version__}, CUDA {torch.version.cuda}")
    print(f"TensorRT {trt.__version__}, ONNX Runtime {ort.__version__}")
    x = casadi.SX.sym("x")
    for backend in ("ipopt", "fatrop"):
        solver = casadi.nlpsol("check", backend, {"x": x, "f": (x - 2) ** 2})
        assert abs(float(solver(x0=0)["x"]) - 2) < 1e-5
    human = erd.human.HumanLoader(height=1.77, weight=62, gender="f").robot
    assert human.model.nq == 43
    symbolic_model = cpin.Model(human.model)
    cpin.framesForwardKinematics(
        symbolic_model, symbolic_model.createData(), casadi.SX.sym("q", human.model.nq))
    assert erd.load("panda").model.nq == 9
    return human


def check_meshcat(human, destination):
    """A synthetic animation checks visualization independently of inference."""
    viewer = meshcat.Visualizer()
    viz = MeshcatVisualizer(human.model, human.collision_model, human.visual_model)
    viz.initViewer(viewer)
    viz.loadViewerModel("installation_check")
    q = pin.neutral(human.model)
    viz.display(q)
    animation = Animation(default_framerate=1)
    current_frame = 0

    def record(node, matrix=np.eye(4)):
        animation.at_frame(node, current_frame).set_transform(matrix)

    with patch.object(Visualizer, "set_transform", record):
        for current_frame in range(3):
            q[0] = 0.05 * current_frame
            viz.display(q)
    assert animation.clips
    viewer.set_animation(animation)
    html = viewer.static_html()
    assert len(html) > 10000 and "set_animation" in html
    destination.write_text(html)
    print(f"CPU Meshcat animation exported to {destination}")


def check_cuda():
    assert torch.cuda.is_available(), "CUDA is required for --gpu"
    print("GPU:", torch.cuda.get_device_name())
    gpu_input = torch.tensor([[-2.0, 3.0]], device="cuda")
    expected = torch.relu(gpu_input)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    tensor = network.add_input("x", trt.float32, (1, 2))
    result = network.add_activation(tensor, trt.ActivationType.RELU).get_output(0)
    result.name = "y"
    network.mark_output(result)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    gpu_output = torch.empty_like(gpu_input)
    assert context.set_tensor_address("x", gpu_input.data_ptr())
    assert context.set_tensor_address("y", gpu_output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(gpu_output, expected)

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Relu", ["x"], ["y"])], "check",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 10
    session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    assert "CUDAExecutionProvider" in session.get_providers(), "ONNX Runtime fell back to CPU"
    output = session.run(None, {"x": gpu_input.cpu().numpy()})[0]
    np.testing.assert_array_equal(output, expected.cpu().numpy())
    print("PyTorch, TensorRT and ONNX Runtime CUDA inference passed")


if __name__ == "__main__":
    human = check_native()
    if sys.argv[1] == "gpu":
        check_cuda()
    else:
        assert not torch.cuda.is_available(), "CPU mode must hide CUDA devices"
        print("GPU inference skipped: CPU-only check")
    check_meshcat(human, Path(sys.argv[2]))

PY_ENV
}

solver() {
  if python3 scripts/python/core/run_ocp_codegen.py --backend both --profile realtime --check; then return; fi
  python3 scripts/python/core/run_ocp_codegen.py --backend both --profile realtime \
    && python3 scripts/python/core/run_ocp_codegen.py --backend both --profile realtime --check
}

ENV_READY=0
if step environment environment; then ENV_READY=1; fi
step tests python3 -m pytest tests/unit -q -p no:cacheprovider || true
step solver solver || true

ROS_READY=0
if [[ -d /root/workspace/ros_ws/src/rtcosmik_ros ]]; then
  if step ros_build bash "${LOGS}/ros.sh" build "${LOGS}"; then ROS_READY=1; fi
else
  skip ros_build 'rtcosmik_ros is not mounted; clone it beside rt-cosmik or set RTCOSMIK_ROS_DIR'
fi

if [[ "${MODE}" == gpu && "${ENV_READY}" == 1 ]]; then
  if step models env BATCHES=4 scripts/bash/fetch_models.sh && step sample scripts/bash/fetch_sample.sh; then
    if step pipeline python3 scripts/python/core/run_pipeline.py \
      --dataset data/comfi_sample --participant 2112 --task RobotWelding --out "${LOGS}/pipeline"; then
      step compare python3 scripts/python/eval/compare_to_mocap.py \
        --reference data/comfi_sample/mocap/aligned/2112/RobotWelding \
        "rt-cosmik=${LOGS}/pipeline" --plots "${LOGS}/eval" || true
    else
      skip compare 'pipeline failed'
    fi
    if [[ "${ROS_READY}" == 1 ]]; then
      step ros_replay bash "${LOGS}/ros.sh" replay "${LOGS}" || true
    else
      skip ros_replay 'overlay is missing or its build failed'
    fi
  else
    skip pipeline 'models or sample unavailable'
    skip compare 'pipeline did not run'
    skip ros_replay 'models or sample unavailable'
  fi
else
  reason='requires --gpu; CPU checks only'
  [[ "${MODE}" == gpu ]] && reason='GPU environment check failed'
  for name in models pipeline compare ros_replay; do skip "${name}" "${reason}"; done
fi
exit "${FAILURES}"
INSIDE

cat > "${LOGS}/ros.sh" <<'ROS'
set -eo pipefail
source /opt/ros/humble/setup.bash
LOGS="$2"
ROS_WS=/tmp/rt-cosmik-ros-check
if [[ "$1" == build ]]; then
  mkdir -p "${ROS_WS}/src"
  # The node writes a scaled URDF into its source tree. Check a private copy.
  cp -r /root/workspace/ros_ws/src/rtcosmik_ros "${ROS_WS}/src/rtcosmik_ros"
  cd "${ROS_WS}"
  colcon build --packages-select rtcosmik_ros
  source install/setup.bash
  /usr/bin/python3 -c 'import rtcosmik_ros.ros2_bridge_node'
  ros2 pkg prefix rviz2
  ros2 pkg prefix robot_state_publisher
  ros2 launch rtcosmik_ros start.launch.py --show-args
  exit
fi
source "${ROS_WS}/install/setup.bash"
# Isolate this replay from other ROS applications on the host.
export ROS_DOMAIN_ID=$((20 + $$ % 180))
# Use a temporary settings module so the test cannot alter the user's subject
# or recording preferences. The sample participant is 1.77 m, 62 kg, female.
python3 - "${LOGS}" <<'PY_SETTINGS'
import pathlib, sys
root=pathlib.Path.cwd()
path=pathlib.Path(sys.argv[1])/'ros_settings.py'
path.write_text(
    'from runpy import run_path\n'
    'BaseSettings=run_path(' + repr(str(root/'settings.py')) + ')["Settings"]\n'
    'class Settings(BaseSettings):\n'
    '    def __post_init__(self):\n'
    '        super().__post_init__()\n'
    '        self.human_height=1.77\n'
    '        self.human_weight=62.0\n'
    '        self.human_gender="f"\n'
    '        self.SAVE_CSV=False\n'
    '        self.SAVE_VID=False\n'
    '        self.record_hotkeys=False\n')
PY_SETTINGS
export RTCOSMIK_SETTINGS_PATH="$(realpath "${LOGS}/ros_settings.py")"
sample="${PWD}/data/comfi_sample"
ros2 launch rtcosmik_ros start.launch.py use_rviz:=false \
  replay_dir:="${sample}/videos/2112/RobotWelding" cam_calib_path:="${sample}/cam_params/2112" \
  > "${LOGS}/ros_launch.log" 2>&1 &
launch_pid=$!
cleanup_ros() {
  kill -INT "${launch_pid}" 2>/dev/null || true
  for _ in $(seq 30); do kill -0 "${launch_pid}" 2>/dev/null || break; sleep 1; done
  kill -TERM "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup_ros EXIT
for _ in $(seq 90); do
  grep -q RTCOSMIK_INIT_DONE "${LOGS}/ros_launch.log" && break
  kill -0 "${launch_pid}" 2>/dev/null || { cat "${LOGS}/ros_launch.log"; exit 1; }
  sleep 1
done
grep -q RTCOSMIK_INIT_DONE "${LOGS}/ros_launch.log"
python3 - <<'PY_ROS'
import math, time
import rclpy
from sensor_msgs.msg import JointState
rclpy.init()
node=rclpy.create_node('rtcosmik_installation_check')
messages=[]
node.create_subscription(JointState, '/rtcosmik/joint_states', messages.append, 10)
end=time.monotonic()+8
while time.monotonic()<end:
    rclpy.spin_once(node, timeout_sec=0.1)
assert len(messages)>1, f'only {len(messages)} joint states received'
assert all(m.position and len(m.name)==len(m.position) and all(map(math.isfinite,m.position)) for m in messages)
assert len({(m.header.stamp.sec,m.header.stamp.nanosec) for m in messages})>1
print(f'{len(messages)} valid joint states received with advancing timestamps')
node.destroy_node()
rclpy.shutdown()
PY_ROS
ROS

STATUS=0
if [[ "${BUILD}" == 1 ]]; then
  echo '[..] build'
  start=$SECONDS
  if "${REPO_ROOT}/docker/run.sh" --cpu --build true > "${LOGS}/build.log" 2>&1; then
    printf 'PASS  %-12s %5ss\n' build "$((SECONDS - start))" | tee "${LOGS}/summary.txt"
  else
    printf 'FAIL  %-12s %5ss  see build.log\n' build "$((SECONDS - start))" | tee "${LOGS}/summary.txt"
    STATUS=1
  fi
else
  if docker image inspect "${RTCOSMIK_IMAGE:-rt-cosmik:latest}" > /dev/null 2>&1; then
    printf 'SKIP  %-12s --no-build\n' build | tee "${LOGS}/summary.txt"
  else
    printf 'FAIL  %-12s requested image is unavailable; remove --no-build\n' build | tee "${LOGS}/summary.txt"
    STATUS=1
  fi
fi
if [[ "${STATUS}" == 0 ]]; then
  if ! "${REPO_ROOT}/docker/run.sh" "${RUN_ARGS[@]}" bash "${LOGS_REL}/inside.sh" "${LOGS_REL}" "${MODE}" \
    2>&1 | tee "${LOGS}/container.log"; then
    STATUS=1
    if ! grep -q '^FAIL' "${LOGS}/summary.txt"; then
      printf 'FAIL  %-12s see container.log\n' container | tee -a "${LOGS}/summary.txt"
    fi
  fi
fi

# The launcher hands checkout files back to the calling user on container exit.
PARENT="$(dirname "${REPO_ROOT}")"
CHECKED=("${REPO_ROOT}")
[[ -d "${PARENT}/cams_calibration" ]] && CHECKED+=("${PARENT}/cams_calibration")
ROS_SOURCE="${RTCOSMIK_ROS_DIR:-${PARENT}/rtcosmik_ros}"
[[ -d "${ROS_SOURCE}" ]] && CHECKED+=("${ROS_SOURCE}")
if find "${CHECKED[@]}" -xdev ! -user "$(id -u)" > "${LOGS}/ownership.log" 2>&1 \
  && [[ ! -s "${LOGS}/ownership.log" ]]; then
  printf 'PASS  %-12s %5ss\n' ownership 0 | tee -a "${LOGS}/summary.txt"
else
  printf 'FAIL  %-12s see ownership.log\n' ownership | tee -a "${LOGS}/summary.txt"
  STATUS=1
fi
if [[ -n "${COPY_TO}" ]]; then
  mkdir -p "${COPY_TO}"
  cp -r "${LOGS}" "${COPY_TO}/"
fi
cat "${LOGS}/summary.txt"
exit "${STATUS}"
