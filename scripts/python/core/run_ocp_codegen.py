import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik.config_loader import settings
from rtcosmik.ik.ik import RT_SWIKA
from rtcosmik.human_model.pin_model import build_dummy_model_no_visuals

human_model = build_dummy_model_no_visuals()

ik_class = RT_SWIKA(human_model, settings.keys_to_track_list, settings.N)
ik_class.compile_Ccode()