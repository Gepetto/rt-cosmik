"""Load a single, import-stable settings module for multiprocessing spawn.

The spawn start method imports modules in fresh child interpreters, so the
settings module must have a canonical import path instead of an ad-hoc dynamic
loader identity.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DefaultSettings:
    """Fallback settings used when project-level settings cannot be imported."""

    SAVE_VID: bool = False
    SAVE_CSV: bool = False
    SAVE_DIR: str = "/default/output/path"

    def __post_init__(self):
        pkg_path = Path(__file__).resolve().parent.parent
        self.cosmik_path = str(pkg_path)
        self.cam_calib_path = str(pkg_path / "config/cam_params")


def load_settings():
    """Load settings from a canonical module path."""
    try:
        # Development / editable install path.
        import settings as settings_module
    except ImportError:
        # Packaged module fallback with canonical identity.
        from rtcosmik import default_settings as settings_module

    try:
        return settings_module.Settings()
    except AttributeError:
        return DefaultSettings()

# Singleton instance accessible throughout the package
settings = load_settings()
