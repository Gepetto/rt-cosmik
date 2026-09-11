#!/usr/bin/env bash
# Fetch the SMPL body models smplfitter needs, then verify the install.
#
# The model files are licence-gated: you must register (free) at each site below
# with the SAME email and password, and the downloader authenticates as you.
# There is no way around this and no redistributable copy.
#
#   https://smpl.is.tue.mpg.de/        (SMPL)
#   https://smpl-x.is.tue.mpg.de/      (SMPL-X)   <- the one this pipeline needs
#   https://mano.is.tue.mpg.de/        (MANO and SMPL+H)
#   https://agora.is.tue.mpg.de/       (kid templates)
#
# Run it interactively and it will prompt:
#
#   bash scripts/bash/setup_smplfitter.sh
#
# Or pass credentials through the environment, so nothing is typed into a shared
# terminal or stored in shell history (note the leading space, which keeps the
# line out of history in bash with HISTCONTROL=ignorespace):
#
#    SMPL_EMAIL='you@example.com' SMPL_PASSWORD='...' bash scripts/bash/setup_smplfitter.sh
#
# The download is ~1 GB and happens once. Point SMPLFITTER_BODY_MODELS at the
# result in your shell profile so it survives a new container.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${SMPLFITTER_BODY_MODELS:-$REPO/weights/body_models}"

python3 -c "import smplfitter" 2>/dev/null || pip install smplfitter

if [ -f "$TARGET/smplx/SMPLX_NEUTRAL.npz" ]; then
    echo "Body models already present at $TARGET"
else
    echo "Downloading body models to $TARGET"
    mkdir -p "$TARGET"
    if [ -n "${SMPL_EMAIL:-}" ] && [ -n "${SMPL_PASSWORD:-}" ]; then
        printf '%s\n%s\n' "$SMPL_EMAIL" "$SMPL_PASSWORD" \
            | python3 -m smplfitter.download "$TARGET"
    else
        python3 -m smplfitter.download "$TARGET"
    fi
fi

echo
echo "Body models at $TARGET (git-ignored)."
echo "settings.body_models_path points here, so no environment variable is needed."
echo
python3 "$REPO/scripts/python/paper/check_smplfitter.py"
