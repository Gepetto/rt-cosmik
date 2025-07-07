import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint



def metrics_par_colonne_df(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """
    Calcule la RMSE pour chaque colonne entre deux DataFrames A et B.
    Retourne une Series indexée par les noms de colonnes.
    """
    assert A.shape == B.shape, "A et B doivent avoir la même taille"
    assert list(A.columns) == list(B.columns), "Les colonnes doivent être identiques"

    corr = A.corrwith(B)
    diff_squared = (A - B) ** 2
    mse = diff_squared.mean(axis=0)       # moyenne sur les lignes
    rmse_rad = np.sqrt(mse)
    rmse_deg = (180 / np.pi) * rmse_rad  # convertir en degrés
    metrics = pd.concat([rmse_deg, corr], axis=1)
    metrics.columns = ['rmse_deg', 'corr']
    return metrics



data_path = sys.argv[1]

dofs  =  ['Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']

excluded_dofs = ['Lwrist_flex_ext', 'Lwrist_x', 'Rwrist_flex_ext', 'Rwrist_x']

upper_dof = ['Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
                          'Lshoulder_flex_ext',
                          'Lshoulder_abd_add', 'Lshoulder_int_ext_rot',
                          'Relbow_flex_ext', 'Relbow_pron_supi', 'Lelbow_flex_ext',
                          'Lelbow_pron_supi']
                          
lower_dof=['Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Lknee_flex_ext', 'Lankle_flex_ext']

for subject in os.listdir(data_path):
    list_of_trials = os.listdir(os.path.join(data_path, subject))
    list_of_trials = [trial for trial in list_of_trials if trial != "cosmik_2cams" and trial != "output_2d" and trial != "results"]
    multi_cols = pd.MultiIndex.from_product([list_of_trials, ['rmse_deg', 'corr']])
    metrics_par_dof_over_trials = pd.DataFrame(index=[dof for dof in dofs if dof not in excluded_dofs].append(["mean", "std"]), columns=multi_cols)
    subject_path = os.path.join(data_path, subject)
    cosmik_2cams_path = os.path.join(data_path, subject, "cosmik_2cams")
    for trial in list_of_trials:
        trial_path = os.path.join(subject_path, trial)
        cosmik_2cams_trial_path = os.path.join(cosmik_2cams_path, trial)
        path_cosmik = os.path.join(cosmik_2cams_trial_path, "q_cosmik_ipopt_2.csv")
        path_mocap = os.path.join(trial_path, "q_mocap_ipopt.csv")

        q_cosmik = pd.read_csv(path_cosmik).iloc[:, 7:]
        q_cosmik.drop(excluded_dofs, axis=1, inplace=True)
        q_mocap = pd.read_csv(path_mocap).iloc[:, 7:]
        q_mocap.drop(excluded_dofs, axis=1, inplace=True)

        # Supprimer la valeur de la dernière ligne si les tailles sont différentes
        if q_cosmik.shape[0] > q_mocap.shape[0]:
            q_cosmik = q_cosmik.iloc[:-1, :]
        elif q_cosmik.shape[0] < q_mocap.shape[0]:
            q_mocap = q_mocap.iloc[:-1, :]

        metrics_par_dof = metrics_par_colonne_df(q_mocap, q_cosmik)
        metrics_par_dof.loc["mean"] = metrics_par_dof.mean()    # ajouter la moyenne
        metrics_par_dof.loc["std"] = metrics_par_dof.std()     # ajouter la variance

        metrics_par_dof_over_trials.loc[:, (trial, "rmse_deg")] = metrics_par_dof["rmse_deg"]
        metrics_par_dof_over_trials.loc[:, (trial, "corr")] = metrics_par_dof["corr"]


    print(metrics_par_dof_over_trials)
    os.makedirs(os.path.join(subject_path, "results"), exist_ok=True)
    metrics_par_dof_over_trials.to_excel(os.path.join(subject_path, "results", f"rmse_par_dof_over_trials_{subject}.xlsx"))

        




