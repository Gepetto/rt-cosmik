import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir

from src.rtcosmik.config_loader import settings
from src.rtcosmik.ik.ik import RT_SWIKA
from src.rtcosmik.human_model.pin_model import build_dummy_model_no_visuals

human_model = build_dummy_model_no_visuals()

ik_class = RT_SWIKA(human_model, settings.keys_to_track_list, settings.N)
ik_class.compile_Ccode()