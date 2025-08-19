import sys
import os


if __name__ == "__main__":
    dataset_path = sys.argv[1]

    subjects = os.listdir(dataset_path)

    for subject in subjects:

        files = os.listdir(os.path.join(dataset_path, subject))

        tasks = [file[:-17] for file in files if file.endswith("_trajectories.csv")]

        base_path = os.path.join(dataset_path, subject)

        print(f"Processing subject {subject}")

        for task in tasks:

            os.makedirs(os.path.join(base_path, task), exist_ok=True)
            path_to_trajectories = os.path.join(base_path, f"{task}_trajectories.csv")
            path_to_jcp = os.path.join(base_path, f"{task}_jcp_mocap.csv")
            path_to_devices = os.path.join(base_path, f"{task}_devices.csv")

            os.rename(path_to_trajectories, os.path.join(base_path, task, f"{task}_trajectories.csv"))
            os.rename(path_to_jcp, os.path.join(base_path, task, f"{task}_jcp_mocap.csv"))
            os.rename(path_to_devices, os.path.join(base_path, task, f"{task}_devices.csv"))
        