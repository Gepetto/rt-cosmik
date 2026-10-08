#!/usr/bin/env bash
# Start RT-COSMIK's container with this checkout mounted, building the image the
# first time.
#
#   docker/run.sh                  # a shell inside the container
#   docker/run.sh <command> ...    # run one command inside it
#   docker/run.sh --build          # rebuild the image first (after a Dockerfile change)
#
# Needs Docker and the NVIDIA Container Toolkit on the host. The container sees
# the GPU, every camera (/dev), and the host network, so the 3D viewer's
# http://127.0.0.1:7000 URL opens in the host's browser.
#
# The companion repositories, cloned next to this one, are mounted too:
#   ../cams_calibration  ->  /root/workspace/cams_calibration
#   ../rtcosmik_ros      ->  /root/workspace/ros_ws/src/rtcosmik_ros
#
# The container runs as root, but what it writes in these directories (models,
# solvers, results) is handed back to you when it exits, so you can edit or
# delete it without sudo. `docker/run.sh true` does only that.
set -euo pipefail

IMAGE="${RTCOSMIK_IMAGE:-rt-cosmik:latest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT=/root/workspace/rt-cosmik

if [[ "${1:-}" == "--build" ]]; then
  shift
  BUILD=1
else
  BUILD=0
fi

if [[ "${BUILD}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[..] building ${IMAGE}; the first build compiles CasADi, Pinocchio and acados (about half an hour)"
  DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" "${REPO_ROOT}/docker"
fi

# OpenCV windows (camera calibration, camera preview) draw on the host's X server.
X11=()
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  if command -v xhost >/dev/null 2>&1; then
    xhost +si:localuser:root >/dev/null || true
  fi
  X11=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

# The container is removed on exit, so companion repositories live on the host.
COMPANIONS=()
MOUNTED=("${MOUNT}")
PARENT="$(dirname "${REPO_ROOT}")"
if [[ -d "${PARENT}/cams_calibration" ]]; then
  COMPANIONS+=(-v "${PARENT}/cams_calibration:/root/workspace/cams_calibration")
  MOUNTED+=(/root/workspace/cams_calibration)
fi
if [[ -d "${PARENT}/rtcosmik_ros" ]]; then
  COMPANIONS+=(-v "${PARENT}/rtcosmik_ros:/root/workspace/ros_ws/src/rtcosmik_ros")
  MOUNTED+=(/root/workspace/ros_ws/src/rtcosmik_ros)
fi

# Files root creates in the mounted directories would be root's on the host too.
# Give them to the calling user when the command ends, normally or on Ctrl-C --
# except with rootless Docker, where root in the container already is that user.
OWNER=()
if ! docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -q rootless; then
  OWNER=(-e "RTCOSMIK_OWNER=$(id -u):$(id -g)" -e "RTCOSMIK_MOUNTED=${MOUNTED[*]}")
fi
read -r -d '' HAND_BACK <<'EOF' || true
hand_back() {
  [[ -n "${RTCOSMIK_OWNER:-}" ]] || return 0
  find ${RTCOSMIK_MOUNTED} -xdev -user 0 -exec chown -h "${RTCOSMIK_OWNER}" {} + 2>/dev/null || true
}
trap 'hand_back; exit 130' INT
trap 'hand_back; exit 143' TERM
"$@"
status=$?
hand_back
exit "${status}"
EOF

TTY=()
[[ -t 0 && -t 1 ]] && TTY=(-it)

# --init and TINI_KILL_PROCESS_GROUP: a Ctrl-C sent to the container reaches the
# command, not only the shell that hands its files back.
exec docker run --rm --init -e TINI_KILL_PROCESS_GROUP=1 "${TTY[@]}" \
  --gpus all --net=host --ipc=host --privileged \
  -v /dev:/dev \
  --device-cgroup-rule "c 81:* rmw" --device-cgroup-rule "c 189:* rmw" \
  "${X11[@]}" "${OWNER[@]}" \
  -v "${REPO_ROOT}:${MOUNT}" "${COMPANIONS[@]}" -w "${MOUNT}" \
  "${IMAGE}" bash -c "${HAND_BACK}" rtcosmik "${@:-bash}"
