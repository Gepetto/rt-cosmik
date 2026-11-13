#!/usr/bin/env python3
"""Build the NLF TensorRT engine that ``settings.nlf_backend = "tensorrt"`` loads.

    python3 scripts/python/core/export_nlf_engine.py

One fp16 engine, written to ``settings.nlf_engine_path``, serves the toolbox
landmark map, the submitted one and the six fold maps. Rebuild it after
changing any landmark map, the NLF weights, TensorRT or the GPU.
"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik import landmark_maps
from rtcosmik.config_loader import settings
from rtcosmik.nlf.crop_network import export_engine


def main():
    vertices = landmark_maps.all_vertices(settings)
    export_engine(settings.nlf_path, settings.cano_path, vertices, settings.nlf_engine_path,
                  device=settings.device)
    print(f"[OK] {settings.nlf_engine_path} ({len(vertices)} vertices)")


if __name__ == "__main__":
    main()
