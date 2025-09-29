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

def rmse(a, b):
    return np.rad2deg(np.sqrt(np.mean((a - b) ** 2)))


data_path = "/home/msabbah/pinocchio-3x/src/rt-cosmik/output"

excluded_dofs = ['Lelbow_pron_supi','Relbow_pron_supi']

# SUBJECTS = ["Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Emmanuelle","Flavie","Guilhem","Herbert","Kahina","Marie_M","Maxime_","Mohamed","Nicolas","Zoe"]

SUBJECTS = ["Alessandro","Bilal","Guilhem"]
TASKS = ["bolting","sanding","overhead","robot_sanding","robot_welding","lifting"]

for subject in SUBJECTS:
    print(subject)
    subject_path = os.path.join(data_path, subject)
    hybrik_path = os.path.join(subject_path, "cosmik_2cams")  # hybrik folder
    mocap_path = os.path.join(subject_path, "mocap")  # mocaps folder

    # Skip subject if hybrik folder does not exist
    if not os.path.exists(hybrik_path):
        print(f"Skipping subject {subject}: 'hybrik' folder not found.")
        continue

    # Skip subject if mocap folder does not exist
    if not os.path.exists(mocap_path):
        print(f"Skipping subject {subject}: 'mocap' folder not found.")
        continue

    list_of_trials = TASKS  # names of motions

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
        hybrik_trials = os.path.join(hybrik_path, trial)
        mocap_trials = os.path.join(mocap_path, trial)
        path_hybrik = os.path.join(hybrik_trials, "q_cosmik_swika.csv")
        path_mocap = os.path.join(mocap_trials, "q_mocap_downsampled.csv")

        # Skip trial if any file is missing
        if not os.path.exists(path_hybrik):
            print(f"Skipping {trial}: hybrik file not found at {path_hybrik}")
            continue
        if not os.path.exists(path_mocap):
            print(f"Skipping {trial}: mocap file not found at {path_mocap}")
            continue

        # read CSVs
        q_hybrik = pd.read_csv(path_hybrik).iloc[:, 7:]
        q_mocap = pd.read_csv(path_mocap).iloc[:, 7:]

        q_hybrik = pd.DataFrame(
                    butterworth_filter(q_hybrik, cutoff_frequency=10.0, order=5, sampling_frequency=40),
                    columns=q_hybrik.columns,
                    index=q_hybrik.index
                )


        # align row counts
        if q_hybrik.shape[0] > q_mocap.shape[0]:
            q_hybrik = q_hybrik.iloc[:-1, :]
        elif q_hybrik.shape[0] < q_mocap.shape[0]:
            q_mocap = q_mocap.iloc[:-1, :]
    

        q_mocap = q_mocap.drop(columns=excluded_dofs, errors='ignore')
        q_hybrik = q_hybrik.drop(columns=excluded_dofs, errors='ignore')

        knee_hybrik = q_hybrik["Rknee_flex_ext"].values
        knee_mocap  = q_mocap["Rknee_flex_ext"].values
        lag = synchronize_signals(knee_hybrik, knee_mocap)
        print("lag",lag)


        if lag > 0:
            # Hybrik delayed → drop first samples of Hybrik
            q_hybrik = q_hybrik[lag:].reset_index(drop=True)
            q_mocap  = q_mocap.iloc[:len(q_hybrik)].reset_index(drop=True)

        elif lag < 0:
            # Mocap delayed → drop first samples of Mocap
            q_mocap  = q_mocap[abs(lag):].reset_index(drop=True)
            q_hybrik = q_hybrik.iloc[:len(q_mocap)].reset_index(drop=True)

        else:
            # No lag
            q_hybrik = q_hybrik.reset_index(drop=True)
            q_mocap  = q_mocap.reset_index(drop=True)

        # plt.plot(q_hybrik["Rknee_flex_ext"].values, label="hybrik")
        # plt.plot(q_mocap["Rknee_flex_ext"].values, label="mocap")
        # plt.legend()
        # plt.show()

        print(f"------------------------RESULTS ARE PRINTED FOR {subject} on {trial} ---------------------------- ")
        print("Lower")

        # Lower limbs flex ext 
        rmse_Rhip_flex = rmse(q_hybrik["Rhip_flex_ext"].values, q_mocap["Rhip_flex_ext"].values)
        rmse_Rknee_flex = rmse(q_hybrik["Rknee_flex_ext"].values, q_mocap["Rknee_flex_ext"].values)
        rmse_Rankle_flex = rmse(q_hybrik["Rankle_flex_ext"].values, q_mocap["Rankle_flex_ext"].values)
        rmse_Lhip_flex = rmse(q_hybrik["Lhip_flex_ext"].values, q_mocap["Lhip_flex_ext"].values)
        rmse_Lknee_flex = rmse(q_hybrik["Lknee_flex_ext"].values, q_mocap["Lknee_flex_ext"].values)
        rmse_Lankle_flex = rmse(q_hybrik["Lankle_flex_ext"].values, q_mocap["Lankle_flex_ext"].values)
        rmse_ll_flex_ext = np.mean(np.array([rmse_Rhip_flex, rmse_Rknee_flex, rmse_Rankle_flex,rmse_Lhip_flex, rmse_Lknee_flex, rmse_Lankle_flex]))
        print(rmse_ll_flex_ext)
        corr_Rhip_flex = np.corrcoef(q_hybrik["Rhip_flex_ext"].values, q_mocap["Rhip_flex_ext"].values)[0,1]
        corr_Rknee_flex = np.corrcoef(q_hybrik["Rknee_flex_ext"].values, q_mocap["Rknee_flex_ext"].values)[0,1]
        corr_Rankle_flex = np.corrcoef(q_hybrik["Rankle_flex_ext"].values, q_mocap["Rankle_flex_ext"].values)[0,1]
        corr_Lhip_flex = np.corrcoef(q_hybrik["Lhip_flex_ext"].values, q_mocap["Lhip_flex_ext"].values)[0,1]
        corr_Lknee_flex = np.corrcoef(q_hybrik["Lknee_flex_ext"].values, q_mocap["Lknee_flex_ext"].values)[0,1]
        corr_Lankle_flex = np.corrcoef(q_hybrik["Lankle_flex_ext"].values, q_mocap["Lankle_flex_ext"].values)[0,1]
        corr_ll_flex_ext = np.mean(np.array([corr_Rhip_flex, corr_Rknee_flex, corr_Rankle_flex, corr_Lhip_flex, corr_Lknee_flex, corr_Lankle_flex]))
        print(corr_ll_flex_ext)

        # Lower limbs abd add
        rmse_Rhip_abd = rmse(q_hybrik["Rhip_abd_add"].values, q_mocap["Rhip_abd_add"].values)
        rmse_Rankle_abd = rmse(q_hybrik["Rankle_abd_add"].values, q_mocap["Rankle_abd_add"].values)
        rmse_Lhip_abd = rmse(q_hybrik["Lhip_abd_add"].values, q_mocap["Lhip_abd_add"].values)
        rmse_Lankle_abd = rmse(q_hybrik["Lankle_abd_add"].values, q_mocap["Lankle_abd_add"].values)
        rmse_ll_abd_add = np.mean(np.array([rmse_Rhip_abd, rmse_Rankle_abd, rmse_Lhip_abd, rmse_Lankle_abd]))
        print(rmse_ll_abd_add)
        corr_Rhip_abd = np.corrcoef(q_hybrik["Rhip_abd_add"].values, q_mocap["Rhip_abd_add"].values)[0,1]
        corr_Rankle_abd = np.corrcoef(q_hybrik["Rankle_abd_add"].values, q_mocap["Rankle_abd_add"].values)[0,1]
        corr_Lhip_abd = np.corrcoef(q_hybrik["Lhip_abd_add"].values, q_mocap["Lhip_abd_add"].values)[0,1]
        corr_Lankle_abd = np.corrcoef(q_hybrik["Lankle_abd_add"].values, q_mocap["Lankle_abd_add"].values)[0,1]
        corr_ll_abd_add = np.mean(np.array([corr_Rhip_abd, corr_Rankle_abd, corr_Lhip_abd, corr_Lankle_abd]))
        print(corr_ll_abd_add)

        # Lower limbs int ext rot
        rmse_Rhip_rot = rmse(q_hybrik["Rhip_int_ext_rot"].values, q_mocap["Rhip_int_ext_rot"].values)
        rmse_Lhip_rot = rmse(q_hybrik["Lhip_int_ext_rot"].values, q_mocap["Lhip_int_ext_rot"].values)
        rmse_ll_int_ext_rot = np.mean(np.array([rmse_Rhip_rot, rmse_Lhip_rot]))
        print(rmse_ll_int_ext_rot)
        corr_Rhip_rot = np.corrcoef(q_hybrik["Rhip_int_ext_rot"].values, q_mocap["Rhip_int_ext_rot"].values)[0,1]
        corr_Lhip_rot = np.corrcoef(q_hybrik["Lhip_int_ext_rot"].values, q_mocap["Lhip_int_ext_rot"].values)[0,1]
        corr_ll_int_ext_rot = np.mean(np.array([corr_Rhip_rot, corr_Lhip_rot]))
        print(corr_ll_int_ext_rot)

        print("Upper")

        #Lumbar flexion extension
        rmse_lumbar_flex = rmse(q_hybrik["Lumbar_flex_ext"].values, q_mocap["Lumbar_flex_ext"].values)
        print(rmse_lumbar_flex)
        corr_lumbar_flex = np.corrcoef(q_hybrik["Lumbar_flex_ext"].values, q_mocap["Lumbar_flex_ext"].values)[0,1]
        print(corr_lumbar_flex)

        #Lumbar lat bend
        rmse_lumbar_lat = rmse(q_hybrik["Lumbar_lateral_flex"].values, q_mocap["Lumbar_lateral_flex"].values)
        print(rmse_lumbar_lat)
        corr_lumbar_lat = np.corrcoef(q_hybrik["Lumbar_lateral_flex"].values, q_mocap["Lumbar_lateral_flex"].values)[0,1]
        print(corr_lumbar_lat)

        #Cervical 
        rmse_cervical_flex = rmse(q_hybrik["Cervical_flex_ext"].values, q_mocap["Cervical_flex_ext"].values)
        rmse_cervical_lat = rmse(q_hybrik["Cervical_lat_bend"].values, q_mocap["Cervical_lat_bend"].values)
        rmse_cervical_rot = rmse(q_hybrik["Cervical_int_ext_rot"].values, q_mocap["Cervical_int_ext_rot"].values)
        rmse_cervical = np.mean(np.array([rmse_cervical_flex, rmse_cervical_lat, rmse_cervical_rot]))
        print(rmse_cervical)
        corr_cervical_flex = np.corrcoef(q_hybrik["Cervical_flex_ext"].values, q_mocap["Cervical_flex_ext"].values)[0,1]
        corr_cervical_lat = np.corrcoef(q_hybrik["Cervical_lat_bend"].values, q_mocap["Cervical_lat_bend"].values)[0,1]
        corr_cervical_rot = np.corrcoef(q_hybrik["Cervical_int_ext_rot"].values, q_mocap["Cervical_int_ext_rot"].values)[0,1]
        corr_cervical = np.mean(np.array([corr_cervical_flex, corr_cervical_lat, corr_cervical_rot]))
        print(corr_cervical)

        #Clavicle 
        rmse_Lcalv = rmse(q_hybrik["Lcalvicule_x"].values, q_mocap["Lcalvicule_x"].values)
        rmse_Rcalv = rmse(q_hybrik["rcalvicule_x"].values, q_mocap["rcalvicule_x"].values)
        rmse_clav = np.mean(np.array([rmse_Lcalv, rmse_Rcalv]))
        print(rmse_clav)
        corr_Lcalv = np.corrcoef(q_hybrik["Lcalvicule_x"].values, q_mocap["Lcalvicule_x"].values)[0,1]
        corr_Rcalv = np.corrcoef(q_hybrik["rcalvicule_x"].values, q_mocap["rcalvicule_x"].values)[0,1]
        corr_clav = np.mean(np.array([corr_Lcalv, corr_Rcalv]))
        print(corr_clav)

        #Shoulders flex ext
        rmse_Rsh_flex = rmse(q_hybrik["Rshoulder_flex_ext"].values, q_mocap["Rshoulder_flex_ext"].values)
        rmse_Lsh_flex = rmse(q_hybrik["Lshoulder_flex_ext"].values, q_mocap["Lshoulder_flex_ext"].values)
        rmse_sh_flex = np.mean(np.array([rmse_Rsh_flex, rmse_Lsh_flex]))
        print(rmse_sh_flex)
        corr_Rsh_flex = np.corrcoef(q_hybrik["Rshoulder_flex_ext"].values, q_mocap["Rshoulder_flex_ext"].values)[0,1]
        corr_Lsh_flex = np.corrcoef(q_hybrik["Lshoulder_flex_ext"].values, q_mocap["Lshoulder_flex_ext"].values)[0,1]
        corr_sh_flex = np.mean(np.array([corr_Rsh_flex, corr_Lsh_flex]))
        print(corr_sh_flex)

        #Shoulders abd add
        rmse_Rsh_abd = rmse(q_hybrik["Rshoulder_abd_add"].values, q_mocap["Rshoulder_abd_add"].values)
        rmse_Lsh_abd = rmse(q_hybrik["Lshoulder_abd_add"].values, q_mocap["Lshoulder_abd_add"].values)
        rmse_sh_abd = np.mean(np.array([rmse_Rsh_abd, rmse_Lsh_abd]))
        print(rmse_sh_abd)
        corr_Rsh_abd = np.corrcoef(q_hybrik["Rshoulder_abd_add"].values, q_mocap["Rshoulder_abd_add"].values)[0,1]
        corr_Lsh_abd = np.corrcoef(q_hybrik["Lshoulder_abd_add"].values, q_mocap["Lshoulder_abd_add"].values)[0,1]
        corr_sh_abd = np.mean(np.array([corr_Rsh_abd, corr_Lsh_abd]))
        print(corr_sh_abd)

        #Shoulders int ext rot
        rmse_Rsh_rot = rmse(q_hybrik["Rshoulder_int_ext_rot"].values, q_mocap["Rshoulder_int_ext_rot"].values)
        rmse_Lsh_rot = rmse(q_hybrik["Lshoulder_int_ext_rot"].values, q_mocap["Lshoulder_int_ext_rot"].values)
        rmse_sh_rot = np.mean(np.array([rmse_Rsh_rot, rmse_Lsh_rot]))
        print(rmse_sh_rot)
        corr_Rsh_rot = np.corrcoef(q_hybrik["Rshoulder_int_ext_rot"].values, q_mocap["Rshoulder_int_ext_rot"].values)[0,1]
        corr_Lsh_rot = np.corrcoef(q_hybrik["Lshoulder_int_ext_rot"].values, q_mocap["Lshoulder_int_ext_rot"].values)[0,1]
        corr_sh_rot = np.mean(np.array([corr_Rsh_rot, corr_Lsh_rot]))
        print(corr_sh_rot)

        #Elbows flex ext
        rmse_Rel_flex = rmse(q_hybrik["Relbow_flex_ext"].values, q_mocap["Relbow_flex_ext"].values)
        rmse_Lel_flex = rmse(q_hybrik["Lelbow_flex_ext"].values, q_mocap["Lelbow_flex_ext"].values)
        rmse_el_flex = np.mean(np.array([rmse_Rel_flex, rmse_Lel_flex]))
        print(rmse_el_flex)
        corr_Rel_flex = np.corrcoef(q_hybrik["Relbow_flex_ext"].values, q_mocap["Relbow_flex_ext"].values)[0,1]
        corr_Lel_flex = np.corrcoef(q_hybrik["Lelbow_flex_ext"].values, q_mocap["Lelbow_flex_ext"].values)[0,1]
        corr_el_flex = np.mean(np.array([corr_Rel_flex, corr_Lel_flex]))
        print(corr_el_flex)

        input(f"End for {subject} on {trial}, press Enter to continue...")
