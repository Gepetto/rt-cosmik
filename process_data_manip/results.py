import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")) # src dir
from src.rtcosmik.config_loader import settings
from src.rtcosmik.utils.read_write_utils import read_mks_data, marker_data_to_dataframe,read_joint_angles_wholebody,read_specific_joint
from src.rtcosmik.utils.linear_algebra_utils import butterworth_filter
from scipy.signal import correlation_lags
from scipy.signal import correlate

def synchronize_signals(sig1, sig2):
    """
    Synchronize two signals by shifting sig2 relative to sig1.

    Args:
        sig1: numpy array, reference signal
        sig2: numpy array, signal to be shifted

    Returns:
        lag: number of samples sig2 was shifted (+ means sig2 delayed)
    """

    corr = correlate(sig1, sig2, mode="full")
    lags = correlation_lags(len(sig1), len(sig2), mode="full")
    lag = lags[np.argmax(corr)]
    return lag

def metrics_par_colonne_df(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la RMSE (en degrés) et la corrélation pour chaque colonne 
    entre deux DataFrames A et B.

    Retourne un DataFrame avec index = noms de colonnes
    et colonnes = ['rmse_deg', 'corr'].
    """
    assert A.shape == B.shape, "A et B doivent avoir la même taille"
    assert list(A.columns) == list(B.columns), "Les colonnes doivent être identiques"

    corr = A.corrwith(B)
    diff_squared = (A - B) ** 2
    mse = diff_squared.mean(axis=0)
    rmse_rad = np.sqrt(mse)
    rmse_deg = rmse_rad * (180 / np.pi)  # un peu plus clair que *180/pi
    metrics = pd.concat([rmse_deg, corr], axis=1)
    metrics.columns = ['rmse_deg', 'corr']
    return metrics




data_path = sys.argv[1]

excluded_dofs = ['Lelbow_pron_supi','Relbow_pron_supi']

for subject in os.listdir(data_path):
    print(subject)
    subject_path = os.path.join(data_path, subject)
    cosmik_2cams_path = os.path.join(subject_path, "cosmik_2cams")  # cosmiks folder
    mocap_path = os.path.join(subject_path, "mocap")  # mocaps folder

    # Skip subject if cosmik_2cams folder does not exist
    if not os.path.exists(cosmik_2cams_path):
        print(f"Skipping subject {subject}: 'cosmik_2cams' folder not found.")
        continue

    # Skip subject if mocap folder does not exist
    if not os.path.exists(mocap_path):
        print(f"Skipping subject {subject}: 'mocap' folder not found.")
        continue

    list_of_trials = os.listdir(cosmik_2cams_path)  # names of motions

    multi_cols = pd.MultiIndex.from_product([list_of_trials, ['rmse_deg', 'corr']])
    metrics_par_dof_over_trials = pd.DataFrame(index=['Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                        #   'thoracic_flex_ext','thoracic_lateral_flex','thoracic_rot_int_ext',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext', #'Lwrist_flex_ext','Lwrist_x',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', #'Rwrist_flex_ext','Rwrist_x',
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add','mean', 'std'], columns=multi_cols)
    
    for trial in list_of_trials:
        print(trial)
        cosmik_trials = os.path.join(cosmik_2cams_path, trial)
        mocap_trials = os.path.join(mocap_path, trial)
        path_cosmik = os.path.join(cosmik_trials, "q_cosmik_swika.csv")
        path_mocap = os.path.join(mocap_trials, "q_mocap_downsampled.csv")

        # Skip trial if any file is missing
        if not os.path.exists(path_cosmik):
            print(f"Skipping {trial}: CoSMIK file not found at {path_cosmik}")
            continue
        if not os.path.exists(path_mocap):
            print(f"Skipping {trial}: mocap file not found at {path_mocap}")
            continue

        # read CSVs
        q_cosmik = pd.read_csv(path_cosmik).iloc[:, 7:]
        q_mocap = pd.read_csv(path_mocap).iloc[:, 7:]

        q_cosmik = pd.DataFrame(
                    butterworth_filter(q_cosmik, cutoff_frequency=10.0, order=5, sampling_frequency=40),
                    columns=q_cosmik.columns,
                    index=q_cosmik.index
                )


        # align row counts
        if q_cosmik.shape[0] > q_mocap.shape[0]:
            q_cosmik = q_cosmik.iloc[:-1, :]
        elif q_cosmik.shape[0] < q_mocap.shape[0]:
            q_mocap = q_mocap.iloc[:-1, :]
    

        q_mocap = q_mocap.drop(columns=excluded_dofs, errors='ignore')
        q_cosmik = q_cosmik.drop(columns=excluded_dofs, errors='ignore')

        knee_cosmik = q_cosmik["Rknee_flex_ext"].values
        knee_mocap  = q_mocap["Rknee_flex_ext"].values
        lag = synchronize_signals(knee_cosmik, knee_mocap)
        print("lag",lag)


        if lag > 0:
            # Cosmik delayed → drop first samples of Cosmik
            q_cosmik = q_cosmik[lag:].reset_index(drop=True)
            q_mocap  = q_mocap.iloc[:len(q_cosmik)].reset_index(drop=True)

        elif lag < 0:
            # Mocap delayed → drop first samples of Mocap
            q_mocap  = q_mocap[abs(lag):].reset_index(drop=True)
            q_cosmik = q_cosmik.iloc[:len(q_mocap)].reset_index(drop=True)

        else:
            # No lag
            q_cosmik = q_cosmik.reset_index(drop=True)
            q_mocap  = q_mocap.reset_index(drop=True)

        # compute metrics
        metrics_par_dof = metrics_par_colonne_df(q_mocap, q_cosmik)
        metrics_par_dof.loc["mean"] = metrics_par_dof.mean()
        metrics_par_dof.loc["std"] = metrics_par_dof.std()

        metrics_par_dof_over_trials.loc[:, (trial, "rmse_deg")] = metrics_par_dof["rmse_deg"]
        metrics_par_dof_over_trials.loc[:, (trial, "corr")] = metrics_par_dof["corr"]

        # Plot each DOF and save figure
        plot_dir = os.path.join(subject_path, "results", "plots_swika_mocap_lag", trial)
        os.makedirs(plot_dir, exist_ok=True)

        for dof in q_cosmik.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(q_mocap[dof], label='Mocap', color='red')
            plt.plot(q_cosmik[dof], label='CoSMIK', color='green')
            plt.title(f"{subject} - {trial} - {dof}")
            plt.xlabel("Frame")
            plt.ylabel("Angle [rad]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{dof}.png"))
            plt.close()
            # plt.show()

    # rmse across all trials
    rmse_mean_over_trials = metrics_par_dof_over_trials.xs('rmse_deg', axis=1, level=1).mean(axis=1)

    metrics_par_dof_over_trials[('mean', 'rmse_deg')] = rmse_mean_over_trials
    corr_mean_over_trials = metrics_par_dof_over_trials.xs('corr', axis=1, level=1).mean(axis=1)
    metrics_par_dof_over_trials[('mean', 'corr')] = corr_mean_over_trials

    # Save 
    results_dir = os.path.join(subject_path, "results")
    os.makedirs(results_dir, exist_ok=True)


    with pd.ExcelWriter(os.path.join(results_dir, f"swika_mocap_lag_rmse_par_dof_over_trials_{subject}.xlsx")) as writer:
        metrics_par_dof_over_trials.to_excel(writer, sheet_name="per_trial")


    




