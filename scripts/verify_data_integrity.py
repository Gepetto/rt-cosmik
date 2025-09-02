import os
import sys
import pandas as pd

dataset_path = sys.argv[1]

threshold = 1

subjects = os.listdir(dataset_path)

for subject in subjects:

    for task in os.listdir(os.path.join(dataset_path, subject)):
        if task == "info.txt":
            continue

        # mks_rt_path = os.path.join(dataset_path, subject, task, f"{task}_mks_rt.csv")
        jcp_hpe_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_hpe.csv")
        jcp_mocap_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_mocap.csv")

        # mks_rt_df = pd.read_csv(mks_rt_path)
        jcp_hpe_df = pd.read_csv(jcp_hpe_path)
        jcp_mocap_df = pd.read_csv(jcp_mocap_path)

        # if abs(mks_rt_df.iloc[1,1]) < 1e-3 or abs(mks_rt_df.iloc[1,1]) > 5:
        #     print(f'{subject}, {task}, mks_rt_df')
        
        # if abs(jcp_hpe_df.iloc[1,1]) < 1e-3 or abs(jcp_hpe_df.iloc[1,1]) > 5:
        #     print(f'{subject}, {task}, jcp_hpe_df')

        difference_df = jcp_hpe_df.sub(jcp_mocap_df)
        
        for column in range(difference_df.shape[1]):
            counter = 0
            for row in range(difference_df.shape[0]):
                if (abs(difference_df.iloc[row, column]) > threshold):
                    counter += 1
                    if counter > 10:
                        print(f"⚠️ Il y a au moins 10 val au-dessus du seuil dans {subject}, {task}, {difference_df.columns[column]}")
                        break