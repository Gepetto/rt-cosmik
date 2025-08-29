import os

# Mapping of subjects to their IDs
subject_ids = {
    "Nicolas": 2307,
    "Mohamed": 1602,
    "Clement": 1118,
    "Mathis": 3361,
    "Claire_": 4827,
    "Anais": 4687,
    "Emmanuelle": 4801,
    "Maxime_": 1847,
    "Alessandro": 4279,
    "Marie_M": 2112,
    "Anastasia": 4216,
    "Flavie": 1012,
    "Zoe": 4162,
    "Kahina": 4665,
    "Herbert": 1508,
    "Guilhem": 4509,
    "Bilal": 4612,
    "Batiste": 2198
}

base_path = "output"

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path) and folder_name in subject_ids:
        new_name = str(subject_ids[folder_name])
        new_path = os.path.join(base_path, new_name)
        os.rename(folder_path, new_path)
        print(f"{folder_name} -> {new_name}")
