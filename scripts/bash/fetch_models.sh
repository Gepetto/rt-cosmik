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

# -------- Detector weights download --------
# All supported detectors are fetched so settings.yolo_model can be changed
# without re-running a download. Measured on real 4-camera frames (batch 4,
# imgsz 640): yolo11n 11.6 ms, yolo26n 14.8 ms, yolov10n 20.1 ms.
# Downloading a checkpoint is a few MB; exporting a TensorRT engine is minutes.
# So fetch every supported detector, but only build engines for the one actually
# selected in settings.py -- otherwise a default install pays 18 exports for
# engines it will never load. Override YOLO_EXPORT_MODELS to build more.
YOLO_MODELS="${YOLO_MODELS:-yolo11n yolo26n yolov10n}"
# Which detector settings.py selects; fall back if the import is unavailable.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
YOLO_SELECTED="$(PYTHONPATH="${SRC_DIR}" python3 -c \
  'from rtcosmik.config_loader import settings; print(settings.yolo_model)' \
  2>/dev/null || echo yolo11n)"
YOLO_EXPORT_MODELS="${YOLO_EXPORT_MODELS:-${YOLO_SELECTED}}"

for MODEL in ${YOLO_MODELS}; do
  MODEL_PT="${YOLO_DIR}/${MODEL}.pt"
  if [[ -f "${MODEL_PT}" ]]; then
    echo "[OK] ${MODEL}.pt already present"
    continue
  fi
  if [[ "${MODEL}" == "yolov10n" ]]; then
    # v10 is not in the ultralytics asset releases.
    download_release_asset "THU-MIG" "yolov10" "v1.1" "yolov10n.pt" "${MODEL_PT}"
  else
    echo "[INFO] Downloading ${MODEL}.pt via ultralytics"
    ( cd "${YOLO_DIR}" && python3 -c "from ultralytics import YOLO; YOLO('${MODEL}.pt')" )
  fi
  [[ -f "${MODEL_PT}" ]] || { echo "[ERR] ${MODEL}.pt not obtained" >&2; exit 2; }
done

# -------- Export TensorRT engines, one per supported camera count --------
# TensorRT engines are built non-dynamic, so an engine's batch size must equal
# the number of cameras fed to it in one call. Rather than tie the install to a
# single rig, build one engine per supported camera count and let the pipeline
# pick the matching one at runtime (see rtcosmik.model_weights).
DEVICE="${DEVICE:-0}"
BATCHES="${BATCHES:-1 2 3 4 5 6}"
IMGSZ=640

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

echo "[INFO] Exporting engines for '${YOLO_EXPORT_MODELS}' (settings.yolo_model), camera counts: ${BATCHES}"
echo "[INFO]   other checkpoints are downloaded but not exported; to build them:"
echo "[INFO]   YOLO_EXPORT_MODELS=\"yolo11n yolo26n\" BATCHES=\"4\" bash scripts/bash/fetch_models.sh"

for MODEL in ${YOLO_EXPORT_MODELS}; do
YOLO_PT="${YOLO_DIR}/${MODEL}.pt"
YOLO_ENGINE="${YOLO_DIR}/${MODEL}.engine"
for BATCH in ${BATCHES}; do
  ENGINE="${YOLO_DIR}/${MODEL}_b${BATCH}.engine"
  META_PATH="${ENGINE}.meta"

  # An existing engine is only reusable if it was built for the same batch and
  # image size; the sidecar records what it was built with.
  if [[ -f "${ENGINE}" ]]; then
    EXISTING_META=""
    [[ -f "${META_PATH}" ]] && EXISTING_META="$(cat "${META_PATH}")"
    if [[ "${EXISTING_META}" == "${BATCH},${IMGSZ}" ]]; then
      echo "[OK] ${MODEL} engine for ${BATCH} camera(s) already present"
      continue
    fi
    echo "[INFO] engine for ${BATCH} camera(s) was built for '${EXISTING_META:-unknown}'; re-exporting."
    rm -f "${ENGINE}" "${META_PATH}"
  fi

  echo "[INFO] Exporting ${MODEL} TensorRT engine for ${BATCH} camera(s) on device=${DEVICE}"
  yolo export \
    model="${YOLO_PT}" \
    format=engine \
    device="${DEVICE}" \
    imgsz=${IMGSZ} \
    batch=${BATCH} \
    dynamic=False \
    simplify=False

  # 'yolo export' writes next to the .pt, so move it to its per-batch name.
  if [[ ! -f "${YOLO_ENGINE}" ]]; then
    echo "[ERR] Export for batch=${BATCH} produced no artifact at ${YOLO_ENGINE}" >&2
    exit 2
  fi
  mv -f "${YOLO_ENGINE}" "${ENGINE}"
  echo "${BATCH},${IMGSZ}" > "${META_PATH}"
  echo "[OK] ${MODEL} engine for ${BATCH} camera(s) saved to: ${ENGINE}"
done
done

echo "[OK] All engines ready in ${YOLO_DIR}"
