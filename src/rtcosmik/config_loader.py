import os
import importlib.util
from pathlib import Path
from dataclasses import dataclass

def load_settings():
    """Load settings from root directory or installed location"""
    try:
        # Try to load from project root (development mode)
        root_path = Path(__file__).resolve().parent.parent.parent
        spec = importlib.util.spec_from_file_location(
            "settings", 
            root_path / "settings.py"
        )
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        return settings.Settings()
    
    except FileNotFoundError:
        # Fallback for installed package
        from dataclasses import dataclass
        @dataclass
        class DefaultSettings:
            # Replicate your settings structure here
            SAVE_VID = False
            SAVE_CSV = False
            SAVE_DIR = "/default/output/path"
            # ... all other fields ...
            
            def __post_init__(self):
                # Calculate paths relative to package location
                pkg_path = Path(__file__).resolve().parent.parent
                self.cosmik_path = str(pkg_path)
                self.cam_calib_path = str(pkg_path / "config/cam_params")
                # ... other path calculations ...

        return DefaultSettings()

# Singleton instance accessible throughout the package
settings = load_settings()