#!/usr/bin/env bash
# Make RT-COSMIK importable everywhere, and optionally build acados.
#
# Two things this fixes, both of which used to be manual:
#   1. rtcosmik on sys.path for any interpreter -- notably the ROS overlay,
#      which lives in another workspace and used to need PYTHONPATH juggling.
#      A .pth in a real site directory is used rather than `pip install -e`,
#      because the system setuptools (59.6) predates PEP 660, and upgrading it
#      risks breaking colcon on ROS 2 Humble.
#   2. acados built and discoverable, for settings.mhe_backend = "acados".
#
# Usage:
#   scripts/bash/setup_env.sh            # path wiring only
#   scripts/bash/setup_env.sh --acados   # also build acados if missing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACADOS_DIR="${ACADOS_DIR:-$(dirname "$REPO_ROOT")/deps/acados}"
BUILD_ACADOS=0
[[ "${1:-}" == "--acados" ]] && BUILD_ACADOS=1

# --- 1. rtcosmik on the path, for every interpreter -------------------------
SITE_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
echo "$REPO_ROOT/src" > "$SITE_DIR/rtcosmik.pth"
echo "[ok] rtcosmik.pth -> $SITE_DIR (points at $REPO_ROOT/src)"
python3 -c "
from rtcosmik.config_loader import settings
print(f'[ok] rtcosmik imports, project root = {settings.cosmik_path}')" 2>/dev/null \
  || { echo '[!!] rtcosmik still not importable'; exit 1; }

# --- 2. acados --------------------------------------------------------------
if [[ "$BUILD_ACADOS" == "1" && ! -f "$ACADOS_DIR/lib/libacados.so" ]]; then
  echo "[..] building acados into $ACADOS_DIR"
  mkdir -p "$ACADOS_DIR"
  # GitHub archives omit submodules, and anonymous git clone is not always
  # available, so the two submodules acados actually needs for its default
  # HPIPM backend are fetched as tarballs.
  if [[ ! -d "$ACADOS_DIR/acados" ]]; then
    curl -sSL https://github.com/acados/acados/archive/refs/heads/main.tar.gz \
      | tar xz -C "$ACADOS_DIR" --strip-components=1
    for repo in giaf/blasfeo giaf/hpipm; do
      name="$(basename "$repo")"
      mkdir -p "$ACADOS_DIR/external/$name"
      curl -sSL "https://github.com/$repo/archive/refs/heads/master.tar.gz" \
        | tar xz -C "$ACADOS_DIR/external/$name" --strip-components=1
    done
  fi
  cmake -S "$ACADOS_DIR" -B "$ACADOS_DIR/build" \
    -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OSQP=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ACADOS_DIR" >/dev/null
  make -C "$ACADOS_DIR/build" -j"$(nproc)" >/dev/null && make -C "$ACADOS_DIR/build" install >/dev/null

  # The python interface pulls in a casadi wheel that would SHADOW a
  # source-built casadi and break pinocchio's casadi bindings, so it is
  # installed without dependencies and the existing casadi is kept.
  pip install -e "$ACADOS_DIR/interfaces/acados_template" --no-deps -q

  # t_renderer renders acados' code-generation templates.
  TERA_VER="$(curl -sSL https://api.github.com/repos/acados/tera_renderer/releases \
              | grep -oE '"tag_name": *"v[0-9.]+"' | head -1 | grep -oE '[0-9.]+')"
  mkdir -p "$ACADOS_DIR/bin"
  curl -sSL -o "$ACADOS_DIR/bin/t_renderer" \
    "https://github.com/acados/tera_renderer/releases/download/v${TERA_VER}/t_renderer-v${TERA_VER}-linux-amd64"
  chmod +x "$ACADOS_DIR/bin/t_renderer"
  echo "[ok] acados built, t_renderer v$TERA_VER"
fi

if [[ -f "$ACADOS_DIR/lib/libacados.so" ]]; then
  cat > /etc/profile.d/acados.sh <<EOF
export ACADOS_SOURCE_DIR=$ACADOS_DIR
export LD_LIBRARY_PATH=$ACADOS_DIR/lib:\${LD_LIBRARY_PATH}
EOF
  chmod +x /etc/profile.d/acados.sh
  echo "[ok] /etc/profile.d/acados.sh -> ACADOS_SOURCE_DIR=$ACADOS_DIR"
  echo "     enable it with settings.mhe_backend = \"acados\""
else
  echo "[--] acados not built (re-run with --acados to build it)"
fi

echo
echo "Done. New shells pick this up automatically; in this one:"
echo "  source /etc/profile.d/acados.sh"
