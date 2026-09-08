#!/usr/bin/env python3
"""Generate the OCP for a marker set other than the one settings.py declares.

The mocap reference tracks 32 markers rather than parity's 35, which is a
structurally different OCP, so it needs its own artefact. run_ocp_codegen builds
whatever settings.py says; this applies a marker set first.

    python3 scripts/python/paper/build_marker_set_ocp.py mocap
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "core"))

from rtcosmik.config_loader import settings
from rtcosmik.paper.mmpose_baseline import apply_marker_set

name = sys.argv[1] if len(sys.argv) > 1 else "mocap"
apply_marker_set(settings, name)
print(f"marker_set={name}: {len(settings.marker_names)} markers, "
      f"{len(settings.locked_joints)} locked DoF")

import run_ocp_codegen as codegen
from rtcosmik.ik import ocp_model

model, keys = codegen.structural_model()
for profile in ("realtime",):
    print(f"  building {settings.mhe_backend}/{profile} -> "
          f"{ocp_model.backend_dir(settings.mhe_backend, settings, profile)}")
    if settings.mhe_backend == "acados":
        codegen.generate_acados(model, keys, profile)
    else:
        codegen.generate_fatrop(model, keys, profile)
print("done")
