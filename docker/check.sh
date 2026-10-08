#!/usr/bin/env bash
# Check the Docker route from end to end, as the README's quick start runs it:
# build the image, then, inside it, check the environment, fetch the models,
# generate the solver, download the sample trial, run it, compare it with motion
# capture, and run the unit tests. When rtcosmik_ros is cloned next to this
# repository, also build it and replay the sample through the ROS 2 node.
# Last, check that everything the container wrote is the caller's again.
#
#   docker/check.sh            # logs in output/docker-check/<date>/
#   docker/check.sh DIR        # and a copy of them in DIR
#
# Needs network access (models and sample are downloaded) and time: the first
# build compiles CasADi, Pinocchio and acados. summary.txt lists each step as
# PASS or FAIL, and <step>.log says why.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOGS_REL="output/docker-check/${RUN_ID}"      # relative to the checkout, inside and out
LOGS="${REPO_ROOT}/${LOGS_REL}"
COPY_TO="${1:-}"
mkdir -p "${LOGS}"

{
  echo "RT-COSMIK Docker check ${RUN_ID}"
  echo "commit: $(git -C "${REPO_ROOT}" log -1 --format='%h %s' 2>/dev/null || echo unknown)"
  echo "host:   $(uname -srm)"
  docker --version
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "nvidia-smi not found"
} > "${LOGS}/host.txt" 2>&1

# -- run inside the container, where the checkout is /root/workspace/rt-cosmik --
cat > "${LOGS}/inside.sh" <<'INSIDE'
set -uo pipefail
cd /root/workspace/rt-cosmik
LOGS="$1"

step() {
  local name="$1" start=$SECONDS
  shift
  echo "[..] ${name}"
  if "$@" > "${LOGS}/${name}.log" 2>&1; then
    printf 'PASS  %-12s %5ss\n' "${name}" "$((SECONDS - start))" | tee -a "${LOGS}/summary.txt"
  else
    printf 'FAIL  %-12s %5ss  see %s.log\n' "${name}" "$((SECONDS - start))" "${name}" \
      | tee -a "${LOGS}/summary.txt"
  fi
}

environment() {
  nvidia-smi && ffmpeg -version | head -1 && v4l2-ctl --version && python3 - <<'PY'
import casadi, cv2, meshcat, tensorrt, torch, ultralytics
import pinocchio, pinocchio.casadi
import acados_template
import example_robot_data as erd
print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available())
print("tensorrt", tensorrt.__version__, "| ultralytics", ultralytics.__version__)
print("pinocchio", pinocchio.__version__, "| casadi", casadi.__version__)
print("human model: nq", erd.human.HumanLoader(height=1.77, weight=62.0, gender="f").robot.model.nq)
print("panda: nq", erd.load("panda").model.nq)
assert torch.cuda.is_available(), "CUDA is not available in the container"
PY
}

step environment environment
step models      env BATCHES=4 scripts/bash/fetch_models.sh
step solver      python3 scripts/python/core/run_ocp_codegen.py --backend acados --profile realtime
step sample      scripts/bash/fetch_sample.sh
step pipeline    python3 scripts/python/core/run_pipeline.py \
                   --dataset data/comfi_sample --participant 2112 --task RobotWelding
step compare     python3 scripts/python/eval/compare_to_mocap.py \
                   --reference data/comfi_sample/mocap/aligned/2112/RobotWelding \
                   rt-cosmik=output/2112/RobotWelding/4cam_mhe_acados --plots "${LOGS}/eval"
step tests       python3 -m pytest tests/unit -q
if [[ -d /root/workspace/ros_ws/src/rtcosmik_ros ]]; then
  step ros_build  bash "${LOGS}/ros.sh" build
  step ros_replay bash "${LOGS}/ros.sh" replay "${LOGS}"
fi
INSIDE

# -- the ROS 2 node: build it, then replay the sample through it --------------
cat > "${LOGS}/ros.sh" <<'ROS'
# ROS setup scripts read unset variables, so no `set -u` here.
source /opt/ros/humble/setup.bash
cd /root/workspace/ros_ws
if [[ "$1" == build ]]; then
  colcon build --packages-select rtcosmik_ros
  exit
fi
LOGS="/root/workspace/rt-cosmik/$2"
source install/setup.bash
sample=/root/workspace/rt-cosmik/data/comfi_sample
timeout -s INT 90 ros2 launch rtcosmik_ros start.launch.py use_rviz:=false \
  replay_dir:=${sample}/videos/2112/RobotWelding cam_calib_path:=${sample}/cam_params/2112 \
  > "${LOGS}/ros_launch.log" 2>&1 &
for _ in $(seq 60); do
  grep -q RTCOSMIK_INIT_DONE "${LOGS}/ros_launch.log" && break
  sleep 1
done
grep "calibrated cameras" "${LOGS}/ros_launch.log"
python3 - <<'PY'
import time
import rclpy
from sensor_msgs.msg import JointState
rclpy.init()
node = rclpy.create_node("rtcosmik_check")
count = [0]
node.create_subscription(JointState, "/rtcosmik/joint_states", lambda msg: count.__setitem__(0, count[0] + 1), 10)
end = time.time() + 10
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"/rtcosmik/joint_states: {count[0]} messages in 10 s")
assert count[0] > 0, "no joint states published"
PY
status=$?
wait
exit ${status}
ROS

echo "[..] build (the first one takes a while)"
start=$SECONDS
if "${REPO_ROOT}/docker/run.sh" --build true > "${LOGS}/build.log" 2>&1; then
  printf 'PASS  %-12s %5ss\n' build "$((SECONDS - start))" | tee "${LOGS}/summary.txt"
  "${REPO_ROOT}/docker/run.sh" bash "${LOGS_REL}/inside.sh" "${LOGS_REL}"
else
  printf 'FAIL  %-12s %5ss  see build.log\n' build "$((SECONDS - start))" | tee "${LOGS}/summary.txt"
fi

# What the container wrote in the checkout and its companions must be yours again.
PARENT="$(dirname "${REPO_ROOT}")"
CHECKED=("${REPO_ROOT}")
for companion in cams_calibration rtcosmik_ros; do
  [[ -d "${PARENT}/${companion}" ]] && CHECKED+=("${PARENT}/${companion}")
done
if find "${CHECKED[@]}" -xdev ! -user "$(id -u)" > "${LOGS}/ownership.log" 2>&1 \
   && [[ ! -s "${LOGS}/ownership.log" ]]; then
  printf 'PASS  %-12s %5ss\n' ownership 0 | tee -a "${LOGS}/summary.txt"
else
  printf 'FAIL  %-12s %5ss  see ownership.log\n' ownership 0 | tee -a "${LOGS}/summary.txt"
fi

if [[ -n "${COPY_TO}" ]]; then
  mkdir -p "${COPY_TO}"
  cp -r "${LOGS}" "${COPY_TO}/"
  echo "Logs copied to ${COPY_TO}/${RUN_ID}"
fi
echo
cat "${LOGS}/summary.txt"
