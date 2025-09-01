import os
import sys
import pandas as pd

dataset_path = sys.argv[1]

subjects = os.listdir(dataset_path)

for subject in subjects:

    for task in os.listdir(os.path.join(dataset_path, subject)):
        if task == "info.txt":
            continue

        mks_rt_path = os.path.join(dataset_path, subject, task, f"{task}_mks_rt.csv")
        jcp_hpe_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_hpe.csv")
        jcp_mocap_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_mocap.csv")

        mks_rt_df = pd.read_csv(mks_rt_path)
        jcp_hpe_df = pd.read_csv(jcp_hpe_path)
        jcp_mocap_df = pd.read_csv(jcp_mocap_path)

        print(f"Checking {task} in {subject} ...")

        # if len(mks_rt_df) > len(jcp_hpe_df) :
        #     mks_rt_df = mks_rt_df.iloc[:-1]
        #     jcp_mocap_df = jcp_mocap_df.iloc[:-1]
        #     mks_rt_df.to_csv(mks_rt_path, index=False)
        #     jcp_mocap_df.to_csv(jcp_mocap_path, index=False)
        # elif len(jcp_hpe_df) > len(mks_rt_df) :
        #     jcp_hpe_df = jcp_hpe_df.iloc[:-1]
        #     jcp_hpe_df.to_csv(jcp_hpe_path, index=False)
        assert len(mks_rt_df) == len(jcp_hpe_df)
        assert len(jcp_mocap_df) == len(jcp_hpe_df)