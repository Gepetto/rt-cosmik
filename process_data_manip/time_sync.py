import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys 
import pandas as pd
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint

def deg2rad(angle):
    return np.pi*angle/180

dofs  =  ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']
dof = ['Rknee_flex_ext']
start_sample =0 
no_trial = "Maxime"
task = "upper"
path_mocap= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/q_mocap_ipopt.csv"
path_cosmik= f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/q_cosmik_ipopt_2.csv"

q_cosmik= read_specific_joint(path_cosmik,dof, start_sample)
q_mocap = read_specific_joint(path_mocap,dof, start_sample)


print(q_mocap.shape,q_cosmik.shape)

corr = correlate(q_mocap,q_cosmik)

lags = correlation_lags(len(q_mocap[:len(q_cosmik)]),len(q_cosmik))

lag = lags[np.argmax(corr)]

print(lag)

if lag >= 0:
    q_mocap_aligned = q_mocap[lag:lag + len(q_cosmik)]
    q_cosmik_aligned = q_cosmik
else:
    q_mocap_aligned = q_mocap[:len(q_cosmik) + lag]
    q_cosmik_aligned = q_cosmik[-lag:]

plt.plot(q_cosmik_aligned, label='cosmik')
plt.plot(q_mocap_aligned, label='mocap')
plt.legend()
plt.title(f"Lag = {lag}")
plt.show()


