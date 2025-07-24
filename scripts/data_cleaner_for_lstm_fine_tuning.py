import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import udp_csv_to_dataframe, read_mks_data, default_mocap_mks_names, read_mmpose_scores_cleaning


if __name__ == "__main__":
    data_to_clean_path = sys.argv[1]
    data_cleaned_path = sys.argv[2]
    seuil = float(sys.argv[3])

    subjects = os.listdir(data_to_clean_path)
    for subject in subjects:
        subject_path = os.path.join(data_to_clean_path, subject)
        scores_dir_path = os.path.join(subject_path, "output_2d")
        cosmik_2cams_path = os.path.join(subject_path, "cosmik_2cams")
        mocap_path = os.path.join(subject_path, "mocap")
        trials = os.listdir(cosmik_2cams_path)
        for trial in trials:
            HPE_data_path = os.path.join(cosmik_2cams_path, trial, "3d_keypoints_filtered_2.csv")
            if "mks_data_gapfilled.csv" in os.listdir(os.path.join(mocap_path, trial)):
                mocap_data_path = os.path.join(mocap_path, trial, "mks_data_gapfilled.csv")
                mocap_data_df = udp_csv_to_dataframe(mocap_data_path, default_mocap_mks_names, udp_type="gapfilled")
            elif "mks_data.csv" in os.listdir(os.path.join(mocap_path, trial)):
                mocap_data_path = os.path.join(mocap_path, trial, "mks_data.csv")
                mocap_data_df = udp_csv_to_dataframe(mocap_data_path, default_mocap_mks_names, udp_type="raw")
            else:
                print(f"Skipping {trial} in {subject} due to missing mocap data.")
                continue
            scores_data_path_cam0 = os.path.join(scores_dir_path, trial, f"{trial}_camera_0.csv")
            scores_data_path_cam2 = os.path.join(scores_dir_path, trial, f"{trial}_camera_2.csv")
            HPE_data_df = pd.read_csv(HPE_data_path)
            scores_data_list = read_mmpose_scores_cleaning([scores_data_path_cam0, scores_data_path_cam2])
            mocap_data_df = mocap_data_df.iloc[:min(len(mocap_data_df), len(HPE_data_df)),:]
            HPE_data_df = HPE_data_df.iloc[:min(len(mocap_data_df), len(HPE_data_df)),:]
            scores_data_list = scores_data_list[:min(len(mocap_data_df), len(HPE_data_df))]

            dropped = 0
            dropped_indexes = [0]
            for ind, scores in enumerate(scores_data_list):
                if min(scores) < seuil:
                    HPE_data_df.drop(HPE_data_df.index[ind-dropped], inplace=True)
                    mocap_data_df.drop(mocap_data_df.index[ind-dropped], inplace=True)
                    dropped += 1
                    dropped_indexes.append(ind-dropped)
                    print(f"Dropped {ind} in {subject}_{trial}")
            dropped_indexes = list(dict.fromkeys(dropped_indexes))

            for num_file, index_dropped in enumerate(dropped_indexes[:-1]):
                if dropped_indexes[num_file + 1] - index_dropped >= 30:
                    os.makedirs(os.path.join(data_cleaned_path, subject, "cosmik_2cams", f"{trial}_{num_file}"), exist_ok=True)
                    os.makedirs(os.path.join(data_cleaned_path, subject, "mocap", f"{trial}_{num_file}"), exist_ok=True)
                    HPE_data_df.iloc[index_dropped:dropped_indexes[num_file + 1]].to_csv(
                        os.path.join(data_cleaned_path, subject, "cosmik_2cams", f"{trial}_{num_file}", "3d_keypoints_filtered_2_cleaned.csv"), index=False)
                    mocap_data_df.iloc[index_dropped:dropped_indexes[num_file + 1]].to_csv(
                        os.path.join(data_cleaned_path, subject, "mocap", f"{trial}_{num_file}", "mks_data_cleaned.csv"), index=False)
            print(f"Cleaned {trial} in {subject}")

            print(f"Cleaned {trial} in {subject}")
