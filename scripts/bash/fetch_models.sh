#!/usr/bin/env bash
set -euo pipefail

# -------- GitHub release asset downloader (generic) --------
get_release_asset_url () {
  local owner="$1" repo="$2" tag="$3" asset="$4"
  python3 - <<PY
import json, os, re, sys, urllib.request
owner="${owner}"; repo="${repo}"; tag="${tag}"; asset="${asset}"
api=f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
headers={"Accept":"application/vnd.github+json","User-Agent":"fetch-assets"}
tok=os.environ.get("GITHUB_TOKEN")
if tok: headers["Authorization"]=f"Bearer {tok}"
req=urllib.request.Request(api, headers=headers)
data=json.load(urllib.request.urlopen(req))
assets=data.get("assets", [])
for a in assets:
    if a.get("name")==asset:
        print(a["browser_download_url"])
        raise SystemExit(0)
print("ERROR: asset not found:", asset, "in", f"{owner}/{repo}@{tag}", file=sys.stderr)
print("Available:", [a.get("name") for a in assets], file=sys.stderr)
raise SystemExit(2)
PY
}

download_release_asset () {
  local owner="$1" repo="$2" tag="$3" asset="$4" out="$5"
  local url
  url="$(get_release_asset_url "$owner" "$repo" "$tag" "$asset")"
  echo "[DL] ${owner}/${repo}@${tag} :: ${asset}"
  mkdir -p "$(dirname "$out")"
  curl -L --retry 5 --retry-delay 2 -o "$out" "$url"
}

# -------- Paths --------
ROOT_DIR="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
WEIGHTS_DIR="${WEIGHTS_DIR:-${ROOT_DIR}/weights}"
NLF_DIR="${NLF_DIR:-${WEIGHTS_DIR}/nlf}"
YOLO_DIR="${YOLO_DIR:-${WEIGHTS_DIR}/yolo}"

mkdir -p "${NLF_DIR}" "${YOLO_DIR}"

# -------- NLF assets --------
NLF_OWNER="isarandi"
NLF_REPO="nlf"
TAG_S="v0.2.2"
ASSET_S="nlf_s_multi_0.2.2.torchscript"
TAG_L="v0.3.2"
ASSET_L="nlf_l_multi_0.3.2.torchscript"

download_release_asset "${NLF_OWNER}" "${NLF_REPO}" "${TAG_S}" "${ASSET_S}" "${NLF_DIR}/${ASSET_S}"
download_release_asset "${NLF_OWNER}" "${NLF_REPO}" "${TAG_L}" "${ASSET_L}" "${NLF_DIR}/${ASSET_L}"

# -------- YOLOv10n weights download --------
# YOLOv10 pretrained weights are in THU-MIG/yolov10 release v1.1: yolov10n.pt :contentReference[oaicite:2]{index=2}
YOLO_OWNER="THU-MIG"
YOLO_REPO="yolov10"
YOLO_TAG="v1.1"
YOLO_ASSET="yolov10n.pt"
YOLO_PT="${YOLO_DIR}/${YOLO_ASSET}"
YOLO_ENGINE="${YOLO_DIR}/yolov10n.engine"

download_release_asset "${YOLO_OWNER}" "${YOLO_REPO}" "${YOLO_TAG}" "${YOLO_ASSET}" "${YOLO_PT}"

# -------- Export to TensorRT engine (fixed imgsz=640, batch=2, not dynamic) --------
DEVICE="${DEVICE:-0}"
BATCH=2
IMGSZ=640


if [[ -f "${YOLO_ENGINE}" ]]; then
  echo "[OK] TensorRT engine already exists: ${YOLO_ENGINE}"
  exit 0
fi

if ! command -v yolo >/dev/null 2>&1; then
  echo "[ERR] 'yolo' CLI not found. Install ultralytics in this environment." >&2
  echo "      pip install ultralytics" >&2
  exit 1
fi

# Quick sanity: exporting to engine typically needs onnx + tensorrt python packages available.
python3 - <<'PY' || true
import importlib
for m in ("onnx","tensorrt"):
    try:
        importlib.import_module(m)
        print("[OK] import", m)
    except Exception as e:
        print("[WARN] cannot import", m, "->", e)
PY

echo "[INFO] Exporting TensorRT engine on device=${DEVICE}"

# Run export and capture logs
LOG="$(mktemp)"
yolo export \
  model="${YOLO_PT}" \
  format=engine \
  device="${DEVICE}" \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  dynamic=False \
  simplify=False | tee "${LOG}"

 

# Double-check that the file built successfully
if [[ ! -f "${YOLO_ENGINE}" ]]; then
  echo "[ERR] Export completed but engine artifact could not be found at ${YOLO_ENGINE}" >&2
  exit 2
fi

# Write sidecar configuration file right next to it as yolov10n.engine.meta
META_PATH="${YOLO_ENGINE}.meta"
echo "${BATCH},${IMGSZ}" > "${META_PATH}"

echo "[OK] Export completed successfully!"
echo "[OK] Engine saved to: ${YOLO_ENGINE}"
echo "[OK] Synchronized verification sidecar metadata layout to: ${META_PATH}"