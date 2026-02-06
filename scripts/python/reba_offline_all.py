import sys
import os
import numpy as np
import pandas as pd
import csv
cosmik_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, cosmik_path) # Repo root
sys.path.insert(0, os.path.join(cosmik_path, "src")) # src dir
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from src.rtcosmik.ergonomics.reba_offline import RebaScore

SUBJECTS = ["Alessandro","Anastasia","Batiste","Bilal","Claire_","Clement"]

# SUBJECTS = [
#      "Alessandro", "Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie","Guilhem","Kahina","Marie_M","Mathis",
#      "Maxime_","Mohamed","Nicolas", "Zoe", "Herbert","Emmanuelle"
# ]

TASKS = ["lifting","overhead","crouch_object","robot_sanding", "robot_welding"]

mocap = False

for subject in SUBJECTS:
    frame_numbers = pd.read_csv('/home/msabbah/pinocchio-3x/src/rt-cosmik/output/ergo/' + subject + '_frames.csv')
    for task in TASKS:
        if mocap:
            data_dir_path = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/mocap/{task}"
        else :
            data_dir_path = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/{subject}/cosmik_2cams/{task}"
        frame_number = frame_numbers[task].values[0]

        if mocap: 
            for file in os.listdir(data_dir_path):
                if "mks_model_mocap.csv" in file:
                    positions_csv_path = os.path.join(data_dir_path, file)
                if "q_mocap.csv" in file:
                    angles_csv_path = os.path.join(data_dir_path, file)
        else: 
            for file in os.listdir(data_dir_path):
                if "mks_model_swika.csv" in file:
                    positions_csv_path = os.path.join(data_dir_path, file)
                if "q_cosmik_swika.csv" in file:
                    angles_csv_path = os.path.join(data_dir_path, file)

        data_angles = pd.read_csv(angles_csv_path)
        data_positions = pd.read_csv(positions_csv_path)

        r_ASIS_study = np.array([data_positions['r.ASIS_study_x'].values[0],
                          data_positions['r.ASIS_study_y'].values[0],
                          data_positions['r.ASIS_study_z'].values[0]])

        L_ASIS_study = np.array([data_positions['L.ASIS_study_x'].values[0],
                          data_positions['L.ASIS_study_y'].values[0],
                          data_positions['L.ASIS_study_z'].values[0]])

        r_shoulder_study = np.array([data_positions['r_shoulder_study_x'].values[0],
                                       data_positions['r_shoulder_study_y'].values[0],
                                       data_positions['r_shoulder_study_z'].values[0]])

        L_shoulder_study = np.array([data_positions['L_shoulder_study_x'].values[0],
                                       data_positions['L_shoulder_study_y'].values[0],
                                       data_positions['L_shoulder_study_z'].values[0]])
        
        right_shoulder_distance_to_rASIS = np.linalg.norm(r_shoulder_study-r_ASIS_study)
        left_shoulder_distance_to_lASIS = np.linalg.norm(L_shoulder_study-L_ASIS_study)

        if task == "lifting":
            reba_object = RebaScore(angles_csv_path, positions_csv_path, frame_number,load_weight=6.5,scaled_disance_rshoulder_to_rASIS=right_shoulder_distance_to_rASIS, scaled_disance_lshoulder_to_lASIS=left_shoulder_distance_to_lASIS)
        else :
            reba_object = RebaScore(angles_csv_path, positions_csv_path, frame_number,scaled_disance_rshoulder_to_rASIS=right_shoulder_distance_to_rASIS, scaled_disance_lshoulder_to_lASIS=left_shoulder_distance_to_lASIS)

        reba_score = reba_object.compute_reba_score()

        print(f"Reba score frame {frame_number} for {subject} on {task}:", reba_score)