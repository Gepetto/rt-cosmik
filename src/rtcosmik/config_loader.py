"""Load a single, import-stable settings module for multiprocessing spawn.

The spawn start method imports modules in fresh child interpreters, so the
settings module must have a canonical import path instead of an ad-hoc dynamic
loader identity.
"""

from dataclasses import dataclass, field
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Optional

LOGGER = logging.getLogger(__name__)


def _guess_project_root() -> Path:
    """Best-effort root discovery for editable and source checkouts."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "settings.py").exists() and (ancestor / "src" / "rtcosmik").exists():
            return ancestor
    for ancestor in here.parents:
        if (ancestor / "config").exists() and (ancestor / "weights").exists():
            return ancestor
    # Fallback for source tree layout: <root>/src/rtcosmik/config_loader.py
    return here.parents[2]


def _find_settings_file() -> Optional[Path]:
    """Locate project-level settings.py if available."""
    env_path = os.getenv("RTCOSMIK_SETTINGS_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        LOGGER.warning("RTCOSMIK_SETTINGS_PATH=%s does not point to a file.", candidate)

    root = _guess_project_root()
    candidate = root / "settings.py"
    if candidate.is_file():
        return candidate
    return None


def _load_module_from_file(module_name: str, module_path: Path) -> Optional[ModuleType]:
    """Import a Python module from an explicit file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        LOGGER.warning("Failed loading module %s from %s: %s", module_name, module_path, exc)
        return None


def _to_spawn_safe_settings(settings_obj):
    """Convert arbitrary settings objects to a pickle-friendly namespace."""
    data = {}
    for name in dir(settings_obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(settings_obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        data[name] = value
    return SimpleNamespace(**data)


@dataclass
class DefaultSettings:
    """Fallback settings used when project-level settings cannot be imported."""

    cosmik_path: str = field(init=False)
    device: str = "cpu"

    # SAVE
    no_trial: str = "default"
    SAVE_VID: bool = False
    SAVE_CSV: bool = False
    record_hotkeys: bool = True
    record_on_start: bool = False
    output_dir: str = field(init=False)  # <repo>/output
    SAVE_DIR: str = field(init=False)    # online trial directory

    # CAMERA
    fs: int = 40
    dt: float = field(init=False)
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"
    cameras: tuple = (0, 2, 4, 6)

    # HUMAN
    human_height: float = 1.81
    human_weight: float = 74.0
    human_gender: str = "m"

    # PATHS
    cam_calib_path: str = field(init=False)
    human_calib_path: str = field(init=False)
    robot_calib_path: str = field(init=False)

    # FILTER
    order: int = 4
    system_freq: int = 40
    cutoff_freq: float = 5.0
    filter_type: str = "lowpass"

    # MODELS
    cano_path: str = field(init=False)
    nlf_path: str = field(init=False)
    yolo_model: str = "yolov10n"
    yolo_path: str = field(init=False)
    yolo_conf: float = 0.2
    yolo_imgsz: int = 640

    # IK
    ik_type: str = "sbs"
    ik_code: str = "python"
    mhe_backend: str = "fatrop"  # "fatrop" (validated reference) or "acados"
    mhe_profile: str = "realtime"  # "realtime" or "accurate"
    cost_weights: list = field(default_factory=lambda: [1, 1e-3, 1e-5])
    N: int = 10
    acados_export_dir: str = None  # default: <repo>/output/acados
    acados_source_dir: str = None  # default: read from ACADOS_SOURCE_DIR env var

    # MARKERS
    marker_names: list = field(default_factory=lambda: [
        "RASI", "LASI", "RPSI", "LPSI",
        "C7", "T11", "T6", "RSHO", "LSHO", "RELB", "LELB", "RMELB", "LMELB", "RWRI", "LWRI", "RMWRI", "LMWRI",
        "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN",
        "RKNE", "LKNE", "RMKNE", "LMKNE", "RANK", "LANK", "RMANK", "LMANK",
        "R5MHD", "L5MHD", "RTOE", "LTOE", "LHEE", "RHEE",
        "Nose", "Head", "REar", "LEar", "REye", "LEye",
    ])
    keys_to_track_list: list = field(default_factory=lambda: [
        "RASI", "LASI", "RPSI", "LPSI",
        "C7", "T11", "T6", "RSHO", "LSHO", "RELB", "LELB", "RMELB", "LMELB", "RWRI", "LWRI", "RMWRI", "LMWRI",
        "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN",
        "RKNE", "LKNE", "RMKNE", "LMKNE", "RANK", "LANK", "RMANK", "LMANK",
        "R5MHD", "L5MHD", "RTOE", "LTOE", "LHEE", "RHEE",
        "Nose", "Head", "REar", "LEar", "REye", "LEye",
    ])
    nlf_indices: list = field(default_factory=lambda: [
        8421, 5727, 8371, 5677,
        5484, 5489, 5500, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721,
        8079, 5361, 7794, 5058, 8022, 5286,
        6401, 3640, 6407, 3646, 8576, 5882, 8680, 8892,
        8474, 5780, 8463, 5770, 8635, 8846,
        9120, 9002, 616, 6, 9929, 9448,
    ])

    joint_angles_names: list = field(default_factory=lambda: [
        'Freeflyer_X[m]', 'Freeflyer_Y[m]', 'Freeflyer_Z[m]', 'Freeflyer_quaternion_X',
        'Freeflyer_quaternion_Y', 'Freeflyer_quaternion_Z', 'Freeflyer_quaternion_W',
        'Left_Hip_Flexion_Extension[rad]', 'Left_Hip_Abduction_Adduction[rad]',
        'Left_Hip_Internal_External_Rotation[rad]', 'Left_Knee_Flexion_Extension[rad]',
        'Left_Ankle_Plantarflexion_Dorsiflexion[rad]', 'Left_Ankle_Inversion_Eversion[rad]',
        'Lumbar_Flexion_Extension[rad]', 'Lumbar_Lateral_Bending[rad]',
        'Thoracic_Flexion_Extension[rad]', 'Thoracic_Lateral_Bending[rad]',
        'Thoracic_Internal_External_Rotation[rad]', 'Left_Clavicle_Elevation_Depression[rad]',
        'Left_Shoulder_Flexion_Extension[rad]', 'Left_Shoulder_Abduction_Adduction[rad]',
        'Left_Shoulder_Internal_External_Rotation[rad]', 'Left_Elbow_Flexion_Extension[rad]',
        'Left_Elbow_Pronation_Supination[rad]', 'Left_Wrist_Flexion_Extension[rad]',
        'Left_Wrist_Radial_Ulnar_Deviation[rad]', 'Cervical_Flexion_Extension[rad]',
        'Cervical_Lateral_Bending[rad]', 'Cervical_Internal_External_Rotation[rad]',
        'Right_Clavicle_Elevation_Depression[rad]', 'Right_Shoulder_Flexion_Extension[rad]',
        'Right_Shoulder_Abduction_Adduction[rad]',
        'Right_Shoulder_Internal_External_Rotation[rad]', 'Right_Elbow_Flexion_Extension[rad]',
        'Right_Elbow_Pronation_Supination[rad]', 'Right_Wrist_Flexion_Extension[rad]',
        'Right_Wrist_Radial_Ulnar_Deviation[rad]', 'Right_Hip_Flexion_Extension[rad]',
        'Right_Hip_Abduction_Adduction[rad]', 'Right_Hip_Internal_External_Rotation[rad]',
        'Right_Knee_Flexion_Extension[rad]', 'Right_Ankle_Plantarflexion_Dorsiflexion[rad]',
        'Right_Ankle_Inversion_Eversion[rad]',
    ])

    def __post_init__(self):
        root = _guess_project_root()
        self.cosmik_path = str(root)
        self.output_dir = str(root / "output")
        self.SAVE_DIR = str(Path(self.output_dir) / self.no_trial)
        self.cam_calib_path = str(root / "config" / "cam_params")
        self.human_calib_path = str(root / "config" / "human_params")
        self.robot_calib_path = str(root / "config" / "robot_params")
        self.cano_path = str(root / "weights" / "canonical_verts" / "smplx.npy")
        self.nlf_path = str(root / "weights" / "nlf" / "nlf_s_multi_0.2.2.torchscript")
        self.yolo_path = str(root / "weights" / "yolo" / f"{self.yolo_model}.engine")
        self.dt = 1 / self.fs


def load_settings():
    """Load settings from a canonical module path."""
    settings_file = _find_settings_file()
    if settings_file is not None:
        settings_module = _load_module_from_file("rtcosmik_project_settings", settings_file)
        if settings_module is not None and hasattr(settings_module, "Settings"):
            try:
                # Settings loaded from an explicit file path are not importable as a
                # normal module in spawned children. Normalize to a stdlib type.
                return _to_spawn_safe_settings(settings_module.Settings())
            except Exception as exc:
                LOGGER.warning("Failed constructing Settings() from %s: %s", settings_file, exc)
        else:
            LOGGER.warning("No Settings class found in %s, falling back to defaults.", settings_file)
    else:
        LOGGER.warning("No project settings.py found; using in-file default settings.")

    return DefaultSettings()

# Singleton instance accessible throughout the package
settings = load_settings()
