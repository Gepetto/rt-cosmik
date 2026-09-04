#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
WORKSPACE_DIR="$(dirname "${ROOT_DIR}")"
FASTSAM_ROOT="${FASTSAM3DBODY_ROOT:-${WORKSPACE_DIR}/deps/Fast-SAM-3D-Body}"
FASTSAM_VENV="${FASTSAM3DBODY_VENV:-${ROOT_DIR}/.venv-fastsam3dbody}"
FASTSAM_URL="https://github.com/kahinachb/Fast-SAM-3D-Body.git"
FASTSAM_COMMIT="9b4ba41a5f1e1175bb870c78f5337a5775a67a88"
MODEL_DIR="${FASTSAM_ROOT}/checkpoints/sam-3d-body-dinov3"
YOLO_DIR="${FASTSAM_ROOT}/checkpoints/yolo"
YOLO_ENGINE="${YOLO_DIR}/yolo11m-pose.engine"
BACKBONE_ENGINE="${MODEL_DIR}/backbone_trt/backbone_dinov3_fp16.engine"

DOWNLOAD_MODELS=0
BUILD_TENSORRT=0
for arg in "$@"; do
    case "${arg}" in
        --download-models) DOWNLOAD_MODELS=1 ;;
        --build-tensorrt) BUILD_TENSORRT=1 ;;
        *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${FASTSAM_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${FASTSAM_ROOT}")"
    git clone "${FASTSAM_URL}" "${FASTSAM_ROOT}"
fi

git -C "${FASTSAM_ROOT}" fetch origin "${FASTSAM_COMMIT}"
git -C "${FASTSAM_ROOT}" checkout --detach "${FASTSAM_COMMIT}"

python3 -m venv "${FASTSAM_VENV}"
export PATH="${FASTSAM_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
"${FASTSAM_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${FASTSAM_VENV}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-fastsam3dbody.txt"
"${FASTSAM_VENV}/bin/python" -m pip install --editable "${ROOT_DIR}"

if [[ "${DOWNLOAD_MODELS}" == 1 ]]; then
    mkdir -p "${YOLO_DIR}"
    (
        cd "${YOLO_DIR}"
        "${FASTSAM_VENV}/bin/python" -c \
            "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
    )

    if ! "${FASTSAM_VENV}/bin/python" -c \
        "from huggingface_hub import HfApi; HfApi().whoami()" \
        >/dev/null 2>&1; then
        echo "The SAM 3D Body checkpoint is gated." >&2
        echo "Accept its license at https://huggingface.co/facebook/sam-3d-body-dinov3" >&2
        echo "and run: ${FASTSAM_VENV}/bin/huggingface-cli login" >&2
        exit 3
    fi

    "${FASTSAM_VENV}/bin/huggingface-cli" download facebook/sam-3d-body-dinov3 \
        --local-dir "${MODEL_DIR}"
fi

if [[ "${BUILD_TENSORRT}" == 1 ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "TensorRT engines must be built inside a GPU-enabled container." >&2
        exit 4
    fi
    test -f "${MODEL_DIR}/model.ckpt"
    test -f "${MODEL_DIR}/assets/mhr_model.pt"
    test -f "${YOLO_DIR}/yolo11m-pose.pt"

    (
        cd "${FASTSAM_ROOT}"
        if [[ ! -f "${YOLO_ENGINE}" ]]; then
            "${FASTSAM_VENV}/bin/python" convert_yolo_pose_trt.py \
                --model "${YOLO_DIR}/yolo11m-pose.pt" --imgsz 640 --half
        fi
        if [[ ! -f "${BACKBONE_ENGINE}" ]]; then
            "${FASTSAM_VENV}/bin/python" convert_backbone_tensorrt.py \
                --export_onnx --convert_trt --batch_sizes 1,3
        fi
    )
fi

cat <<EOF
Fast SAM 3D Body environment ready.

Source: ${FASTSAM_ROOT}
Python: ${FASTSAM_VENV}/bin/python

Run the integration test with:
FASTSAM3DBODY_ROOT=${FASTSAM_ROOT} \\
${FASTSAM_VENV}/bin/python -m pytest -s \\
  ${ROOT_DIR}/tests/unit/pose_estimator/test_fastsam3dbody_comfi.py
EOF
