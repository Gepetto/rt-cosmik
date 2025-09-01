import os
import sys
import pandas as pd

dataset_path = sys.argv[1]

subjects = os.listdir(dataset_path)

counter_samples = 0
counter_trials = 0

for subject in subjects:

    for task in os.listdir(os.path.join(dataset_path, subject)):
        if task == "info.txt":
            continue

        mks_rt_path = os.path.join(dataset_path, subject, task, f"{task}_mks_rt.csv")

        mks_rt_df = pd.read_csv(mks_rt_path)

        counter_samples += len(mks_rt_df)
        counter_trials += 1

print(f"Average number of samples per trial: {counter_samples/counter_trials}")





