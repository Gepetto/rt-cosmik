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


data_dir_path = sys.argv[1]
frame_number = int(sys.argv[2])

for file in os.listdir(data_dir_path):
    if "mks_model_mocap.csv" in file:
        positions_csv_path = os.path.join(data_dir_path, file)
    if "q_mocap.csv" in file:
        angles_csv_path = os.path.join(data_dir_path, file)

data_angles = pd.read_csv(angles_csv_path)
data_positions = pd.read_csv(positions_csv_path)

if frame_number == 0:

    with open(os.path.join(data_dir_path, "reba_score.csv"), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["frame_number", "reba_score"])

        for i in range(data_angles.shape[0]):

            if i == 0:
                reba_object = RebaScore(angles_csv_path, positions_csv_path, i)
                scaled_disance_rshoulder_to_rASIS = reba_object.upper_arms["right_shoulder_distance_to_rASIS"]
                scaled_disance_lshoulder_to_lASIS = reba_object.upper_arms["left_shoulder_distance_to_lASIS"]
                print(scaled_disance_lshoulder_to_lASIS, scaled_disance_rshoulder_to_rASIS)
            else :
                reba_object = RebaScore(angles_csv_path, positions_csv_path, i, scaled_disance_rshoulder_to_rASIS=scaled_disance_rshoulder_to_rASIS, 
                                        scaled_disance_lshoulder_to_lASIS=scaled_disance_lshoulder_to_lASIS)

            reba_score = reba_object.compute_reba_score()

            writer.writerow([i, reba_score])

            print(f"Reba score frame {i}:", reba_score)

else:

    reba_object = RebaScore(angles_csv_path, positions_csv_path, frame_number)

    reba_score = reba_object.compute_reba_score()

    print(f"Reba score frame {frame_number}:", reba_score)