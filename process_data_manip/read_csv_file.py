import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#data = pd.read_csv('responses_all_conc.csv', header=None)

headers = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']


def csv_to_dict_of_dicts(df, headers):
    result = {}
    for i, header in enumerate(headers):
        # Each header corresponds to three consecutive columns (x, y, z)
        x_col = df.iloc[:, i*3]
        y_col = df.iloc[:, i*3 + 1]
        z_col = df.iloc[:, i*3 + 2]
        
        # Store x, y, z values as lists in the dictionary
        result[header] = {
            'x': x_col.tolist(),
            'y': y_col.tolist(),
            'z': z_col.tolist()
        }
    return result

# dict_of_dicts = csv_to_dict_of_dicts(data, headers)
# print(dict_of_dicts)