import pandas as pd
import numpy as np
from src.rtcosmik.config_loader import settings
import sys

data_path = sys.argv[1]

dofs  = settings.joint_angles_names

upper_dof = ['Lumbar_flex_ext', 'Lumbar_int_ext_rot',
            'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
            'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
            'Lshoulder_flex_ext',
            'Lshoulder_abd_add', 'Lshoulder_int_ext_rot',
            'Relbow_flex_ext', 'Relbow_pron_supi', 'Lelbow_flex_ext',
            'Lelbow_pron_supi']
                          
lower_dof = ['Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Lhip_flex_ext', 'Lhip_abd_add', 
            'Lhip_int_ext_rot',
            'Rknee_flex_ext','Rankle_flex_ext', 'Lknee_flex_ext', 'Lankle_flex_ext']

for i, name in enumerate(dof):



