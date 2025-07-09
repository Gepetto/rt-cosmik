import numpy as np
import pandas as pd
import sys
import os
import csv

def align_points(P_cam, P_mocap):
    """
    Calcule la rotation et la translation qui alignent les points caméras sur les points mocap.
    """
    assert P_cam.shape == P_mocap.shape, "Les deux ensembles doivent avoir le même nombre de points."

    # Erreur RMS avant alignement (dans les repères d'origine)
    initial_errors = np.linalg.norm(P_cam - P_mocap, axis=1)
    initial_rms_error = np.sqrt(np.mean(initial_errors**2))
    # print("Erreur RMS avant alignement :", initial_rms_error)

    # Centrage des deux ensembles
    centroid_cam = P_cam.mean(axis=0)
    centroid_mocap = P_mocap.mean(axis=0)
    P_cam_centered = P_cam - centroid_cam
    P_mocap_centered = P_mocap - centroid_mocap

    # Matrice de covariance
    H = P_mocap_centered.T @ P_cam_centered

    # Décomposition SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Translation
    t = centroid_mocap - R @ centroid_cam

    # Points transformés
    P_cam_aligned = (R @ P_cam.T).T + t

    # Erreur de reconstruction
    errors = np.linalg.norm(P_cam_aligned - P_mocap, axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    return R, t, rms_error, P_cam_aligned, initial_rms_error

# === Exemple d'utilisation avec tes données ===
data_path = sys.argv[1]

marker_mocap_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis']

lstm_mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']


markers_to_compare = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study',
           'r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study']


subjects = os.listdir(data_path)
for subject in subjects:
    subject_path = os.path.join(data_path, subject)
    trials = os.listdir(subject_path)
    results_path = os.path.join(subject_path, "results")
    for trial in trials:
        if trial == "results":
            continue
        trial_path = os.path.join(subject_path, trial)
        data_list = os.listdir(trial_path)
        if "mks_data_gapfilled.csv" in data_list:
            mocap_data_path = os.path.join(trial_path, "mks_data_gapfilled.csv")
            with open(mocap_data_path) as f:
                reader = csv.reader(f)
                raw_mocap_data = np.array([row for row in reader])  # Charger les données
                raw_mocap_data = raw_mocap_data[:,2:].astype(float)  # Convertir en float et ignorer la première colonne (timestamps)
                P_mocap = np.array(raw_mocap_data).reshape(-1, 53, 3)
        else:
            mocap_data_path = os.path.join(trial_path, "mks_data.csv")
            with open(mocap_data_path) as f:
                reader = csv.reader(f)
                header = next(reader)  # Lire l'en-tête
                raw_mocap_data = np.array([row for row in reader])  # Charger les données
                raw_mocap_data = raw_mocap_data[:,1:].astype(float)  # Convertir en float et ignorer la première colonne (timestamps)
                P_mocap = np.array(raw_mocap_data).reshape(-1, 53, 3)
        
        P_cam = np.array(pd.read_csv(os.path.join(trial_path, "augmented_markers_2.csv")).values).reshape(-1, 43,3)  # ou autre méthode de chargement
        
        list_R = []
        list_t = []
        list_rms_error = []
        list_P_cam_aligned = []
        list_initial_rms_error = []

        for i in range(min(P_mocap.shape[0], P_cam.shape[0])):
            P_cam_current = pd.DataFrame(P_cam[i,:,:], index=lstm_mks_names, columns=['x', 'y', 'z'])
            P_mocap_current = pd.DataFrame(P_mocap[i,:,:], index=marker_mocap_names, columns=['x', 'y', 'z'])
            P_cam_current = P_cam_current.loc[markers_to_compare]
            P_mocap_current = P_mocap_current.loc[markers_to_compare]
            P_cam_current = P_cam_current.values
            P_mocap_current = P_mocap_current.values

            R, t, rms_error, P_cam_aligned_current, initial_rms_error = align_points(P_cam_current, P_mocap_current)
            # print("Rotation R:\n", R)
            # print("Translation t:\n", t)
            # print("Erreur RMS (alignement):", rms_error)
            list_R.append(R)
            list_t.append(t)
            list_rms_error.append(rms_error)
            list_P_cam_aligned.append(P_cam_aligned_current)
            list_initial_rms_error.append(initial_rms_error)

        R_mean = np.mean(list_R, axis=0)
        R_std = np.std(list_R, axis=0)
        t_mean = np.mean(list_t, axis=0)
        t_std = np.std(list_t, axis=0)
        rms_mean = np.mean(list_rms_error)
        rms_std = np.std(list_rms_error)
        initial_rms_error_mean = np.mean(list_initial_rms_error)
        initial_rms_error_std = np.std(list_initial_rms_error)
        print("Trial :", trial)
        print("Rotation moyenne R:\n", R_mean)
        print("Rotation std:\n", R_std)
        print("Translation moyenne t:\n", t_mean)
        print("Translation std:\n", t_std)
        print("rms error mean :", rms_mean)
        print("rms error std :", rms_std)
        print("Initial RMS error mean :", initial_rms_error_mean)
        print("Initial RMS error std :", initial_rms_error_std)

        # Sauvegarder les résultats dans un fichier CSV
        output_file = os.path.join(results_path, f"Procrustes_alignment_results_{trial}.csv")

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["R_mean"])
            for row in R_mean:
                writer.writerow(row)
            writer.writerow(["R_std"])
            for row in R_std:
                writer.writerow(row)
            writer.writerow(["t_mean"] + t_mean.tolist())
            writer.writerow(["t_std"] + t_std.tolist())
            writer.writerow(["rms_mean", rms_mean])
            writer.writerow(["rms_std", rms_std])
            writer.writerow(["initial_rms_error_mean", initial_rms_error_mean])
            writer.writerow(["initial_rms_error_std", initial_rms_error_std])

