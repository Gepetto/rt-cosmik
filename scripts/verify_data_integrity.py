import os
import sys
import pandas as pd
import shutil
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rtcosmik.utils.read_write_utils import read_mks_data, default_mocap_mks_names, read_subject_info

dataset_path = sys.argv[1]
dst_dir = sys.argv[2]
mode = sys.argv[3]

threshold = 1

subjects = os.listdir(dataset_path)

for subject in subjects:

    for task in os.listdir(os.path.join(dataset_path, subject)):
        if mode == "correcter":
            os.makedirs(os.path.join(dst_dir, subject, task), exist_ok=True)
            if task == "info.txt":
                shutil.copy2(os.path.join(dataset_path, subject, task), os.path.join(dst_dir, subject, task))
                continue

            mks_rt_path = os.path.join(dataset_path, subject, task, f"{task}_mks_rt.csv")
            jcp_hpe_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_hpe.csv")
            jcp_mocap_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_mocap.csv")

            mks_rt_df = pd.read_csv(mks_rt_path)
            jcp_hpe_df = pd.read_csv(jcp_hpe_path)
            jcp_mocap_df = pd.read_csv(jcp_mocap_path)

            # if abs(mks_rt_df.iloc[1,1]) < 1e-3 or abs(mks_rt_df.iloc[1,1]) > 5:
            #     print(f'{subject}, {task}, mks_rt_df')
            
            # if abs(jcp_hpe_df.iloc[1,1]) < 1e-3 or abs(jcp_hpe_df.iloc[1,1]) > 5:
            #     print(f'{subject}, {task}, jcp_hpe_df')

            difference_df = jcp_hpe_df.sub(jcp_mocap_df)
            
            first_index_bug = 100000000
            for column in range(difference_df.shape[1]):
                for row in range(difference_df.shape[0]):
                    if (abs(difference_df.iloc[row, column]) > threshold):
                        if row < first_index_bug:
                            first_index_bug = row
            
            if first_index_bug != 100000000:
                print(f'⚠️ {subject}, {task}, first index bug: {first_index_bug}')
                mks_rt_df = mks_rt_df.iloc[:first_index_bug]
                jcp_hpe_df = jcp_hpe_df.iloc[:first_index_bug]
                jcp_mocap_df = jcp_mocap_df.iloc[:first_index_bug]
            else:
                print(f'{subject}, {task}, no bug')


            mks_rt_df.to_csv(os.path.join(dst_dir, subject, task, f"{task}_mks_rt.csv"), index=False)
            jcp_hpe_df.to_csv(os.path.join(dst_dir, subject, task, f"{task}_jcp_hpe.csv"), index=False)
            jcp_mocap_df.to_csv(os.path.join(dst_dir, subject, task, f"{task}_jcp_mocap.csv"), index=False)
        
        elif mode == "verifier":
            if task == "info.txt":
                continue
            mks_rt_path = os.path.join(dst_dir, subject, task, f"{task}_mks_rt.csv")
            jcp_hpe_path = os.path.join(dst_dir, subject, task, f"{task}_jcp_hpe.csv")
            jcp_mocap_path = os.path.join(dst_dir, subject, task, f"{task}_jcp_mocap.csv")

            mks_rt_df = pd.read_csv(mks_rt_path)
            jcp_hpe_df = pd.read_csv(jcp_hpe_path)
            jcp_mocap_df = pd.read_csv(jcp_mocap_path)

            difference_df = jcp_hpe_df.sub(jcp_mocap_df)
            
            for column in range(difference_df.shape[1]):
                for row in range(difference_df.shape[0]):
                    if (abs(difference_df.iloc[row, column]) > threshold):
                        print(f'{subject}, {task}, {row}')
        
        elif mode == "test_verifier":
            if subject in ["Kahina", "Flavie"]:
                if task in ["bolting", "sanding", "overhead", "robot_sanding", "robot_welding", "bolting_sat", "lifting"]:
                    mks_rt_path = os.path.join(dst_dir, subject, task, f"{task}_mks_rt.csv")
                    jcp_hpe_path = os.path.join(dst_dir, subject, task, f"{task}_jcp_hpe.csv")
                    jcp_mocap_path = os.path.join(dst_dir, subject, task, f"{task}_jcp_mocap.csv")

                    mks_rt_df = pd.read_csv(mks_rt_path)
                    jcp_hpe_df = pd.read_csv(jcp_hpe_path)
                    jcp_mocap_df = pd.read_csv(jcp_mocap_path)

                    difference_df = jcp_hpe_df.sub(jcp_mocap_df)

                    list_of_row_that_bug = []
                    
                    for column in range(difference_df.shape[1]):
                        for row in range(difference_df.shape[0]):
                            if (abs(difference_df.iloc[row, column]) > threshold) and (row not in list_of_row_that_bug):
                                list_of_row_that_bug.append(row)
                    
                    print(f'{subject}, {task}, {list_of_row_that_bug}')

        elif mode == "midhip_correcter":
            # os.makedirs(os.path.join(dst_dir, subject, task), exist_ok=True)
            if task == "info.txt":
                # shutil.copy2(os.path.join(dataset_path, subject, task), os.path.join(dst_dir, subject, task))
                continue

            mks_rt_path = os.path.join(dataset_path, subject, task, f"{task}_mks_rt.csv")
            jcp_hpe_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_hpe.csv")
            jcp_mocap_path = os.path.join(dataset_path, subject, task, f"{task}_jcp_mocap.csv")

            mks_rt_df = pd.read_csv(mks_rt_path)
            jcp_hpe_df = pd.read_csv(jcp_hpe_path)
            jcp_mocap_df = pd.read_csv(jcp_mocap_path)

            k_list, _ = read_mks_data(jcp_hpe_df, converter=1)        # includes 'midHip'
            m_list, _ = read_mks_data(jcp_mocap_df, converter=1)

            mid_hpe = np.zeros((len(k_list), 3), dtype=np.float32)
            for i, fr in enumerate(k_list):
                mid_hpe[i] = fr['midHip']
                
            mid = np.zeros((len(k_list), 3), dtype=np.float32)
            for i, fr in enumerate(k_list):
                mid[i] = fr['midHip']

            jcp_hpe_np = jcp_hpe_df.to_numpy()
            jcp_mocap_np = jcp_mocap_df.to_numpy()

            jcp_hpe_np = jcp_hpe_np.reshape(len(k_list), 20, 3)
            jcp_mocap_np = jcp_mocap_np.reshape(len(k_list), 20, 3)

            mid_hpe = mid_hpe[:, None, :]
            mid = mid[:, None, :]

            jcp_hpe_sub_np = jcp_hpe_np - mid_hpe
            jcp_mocap_sub_np = jcp_mocap_np - mid

            jcp_hpe_sub_np = jcp_hpe_sub_np.reshape(len(k_list), 20*3)
            jcp_mocap_sub_np = jcp_mocap_sub_np.reshape(len(k_list), 20*3)

            jcp_hpe_df = pd.DataFrame(jcp_hpe_sub_np, columns=jcp_hpe_df.columns)
            jcp_mocap_df = pd.DataFrame(jcp_mocap_sub_np, columns=jcp_mocap_df.columns)

            # if abs(mks_rt_df.iloc[1,1]) < 1e-3 or abs(mks_rt_df.iloc[1,1]) > 5:
            #     print(f'{subject}, {task}, mks_rt_df')
            
            # if abs(jcp_hpe_df.iloc[1,1]) < 1e-3 or abs(jcp_hpe_df.iloc[1,1]) > 5:
            #     print(f'{subject}, {task}, jcp_hpe_df')

            difference_df = jcp_hpe_df.sub(jcp_mocap_df)
            
            # first_index_bug = 100000000
            for column in range(difference_df.shape[1]):
                for row in range(difference_df.shape[0]):
                    if (abs(difference_df.iloc[row, column]) > threshold):
                        print(f'{subject}, {task}, {row}')
