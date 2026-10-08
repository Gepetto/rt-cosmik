#!/usr/bin/env bash
# Download a sample trial to try RT-COSMIK without cameras.
#
#   scripts/bash/fetch_sample.sh          # into data/comfi_sample
#   scripts/bash/fetch_sample.sh DIR      # into DIR/comfi_sample
#
# 20 s of one COMFI recording (participant 2112, RobotWelding, cameras 0 2 4 6)
# with its calibration, subject, robot states and motion capture reference, in
# the dataset layout run_pipeline.py reads. 66 MB. COMFI is CC BY 4.0:
# https://doi.org/10.5281/zenodo.17223909
set -euo pipefail

TAG="sample-data-v1"
ASSET="comfi_sample_2112_RobotWelding.tar.gz"
SHA256="e21d5ad3adc0d6d9eb4bc08d4ef279c55ce11cd2839e48e789ca342c0d36ed65"
URL="https://github.com/Gepetto/rt-cosmik/releases/download/${TAG}/${ASSET}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-${ROOT_DIR}/data}"
SAMPLE="${DEST}/comfi_sample"

if [[ -d "${SAMPLE}/videos" ]]; then
  echo "[OK] sample already in ${SAMPLE}"
else
  mkdir -p "${DEST}"
  ARCHIVE="$(mktemp)"
  trap 'rm -f "${ARCHIVE}"' EXIT
  echo "[DL] ${URL}"
  curl -fL --retry 5 --retry-delay 2 -o "${ARCHIVE}" "${URL}"
  if ! echo "${SHA256}  ${ARCHIVE}" | sha256sum --check --status; then
    echo "[ERR] ${ASSET} does not match its checksum; download it again." >&2
    exit 2
  fi
  tar -xzf "${ARCHIVE}" -C "${DEST}"
  echo "[OK] sample extracted to ${SAMPLE}"
fi

# Print the path as the user would type it from the repository root.
SHOWN="${SAMPLE#"${ROOT_DIR}/"}"
cat <<EOF

Run RT-COSMIK on it:
  python3 scripts/python/core/run_pipeline.py --dataset ${SHOWN} --participant 2112 --task RobotWelding
EOF
