"""The fallback settings must match settings.py.

DefaultSettings is used when settings.py cannot be found or loaded. Its values
once drifted from settings.py (another IK, another filter cutoff, the heels
swapped against nlf_indices), so a broken settings.py silently ran a different
pipeline. This test keeps the two in step.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from rtcosmik.config_loader import DefaultSettings      # noqa: E402

SETTINGS_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "settings.py"))


def test_default_settings_match_settings_py():
    spec = importlib.util.spec_from_file_location("project_settings", SETTINGS_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project, fallback = module.Settings(), DefaultSettings()

    names = {n for n in dir(project) + dir(fallback) if not n.startswith("_")}
    names.discard("device")  # settings.py picks CUDA; the fallback avoids torch
    for name in sorted(names):
        assert getattr(fallback, name, None) == getattr(project, name, None), name
