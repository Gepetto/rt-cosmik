#!/usr/bin/env bash
# Fetch the OpenCap v0.3 LSTM marker augmenter used by the mmpose baseline.
#
# The weights live in this repository's own history, on the full_model branch,
# where they were committed before the *.onnx ignore rule existed. They are not
# re-committed here: the repo keeps model weights out of the tree and fetches
# them (see fetch_models.sh), and this extraction needs no network.
#
# The regular v0.3 model is used, not model_finetuned.onnx -- the finetuned
# variant was trained on other data and is not what the original pipeline ran.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_REF="${AUGMENTER_REF:-origin/full_model}"
DEST="${REPO_ROOT}/src/rtcosmik/augmenter/augmentation_model/LSTM"

cd "${REPO_ROOT}"
if ! git rev-parse --verify --quiet "${SRC_REF}" >/dev/null; then
  echo "[ERR] ${SRC_REF} not found. Fetch it first: git fetch origin full_model" >&2
  exit 1
fi

for PART in lower upper; do
  DIR="${DEST}/v0.3_${PART}"
  mkdir -p "${DIR}"
  # augmentTRC also normalises its inputs with the training mean/std.
  for FILE in model.onnx metadata.json mean.npy std.npy; do
    SRC_PATH="src/rtcosmik/augmenter/augmentation_model/LSTM/v0.3_${PART}/${FILE}"
    if [[ -s "${DIR}/${FILE}" ]]; then
      echo "[OK] v0.3_${PART}/${FILE} already present"
      continue
    fi
    if ! git cat-file -e "${SRC_REF}:${SRC_PATH}" 2>/dev/null; then
      echo "[ERR] ${SRC_PATH} missing from ${SRC_REF}" >&2
      exit 2
    fi
    git show "${SRC_REF}:${SRC_PATH}" > "${DIR}/${FILE}"
    echo "[OK] extracted v0.3_${PART}/${FILE} ($(stat -c%s "${DIR}/${FILE}") bytes)"
  done
done

python3 - "${DEST}" <<'PYEOF'
import sys, onnxruntime as ort
dest = sys.argv[1]
for part, feat in (("lower", 47), ("upper", 23)):
    s = ort.InferenceSession(f"{dest}/v0.3_{part}/model.onnx")
    i = s.get_inputs()[0]
    assert i.shape[-1] == feat, f"v0.3_{part} expects {feat} features, got {i.shape[-1]}"
    print(f"[OK] v0.3_{part} loads: {i.shape} -> {s.get_outputs()[0].shape}")
PYEOF
echo "[OK] augmenter ready"
