import yaml
import glob
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

# Folder where all subjects have their info.txt
base_path = "/root/workspace/ros_ws/src/rt-cosmik/output"
output_folder = "/root/workspace/ros_ws/src/rt-cosmik/output/metadata"
os.makedirs(output_folder, exist_ok=True)

for subject, subj_id in subject_ids.items():
    info_file = os.path.join(base_path, subject, "info.txt")
    if not os.path.exists(info_file):
        print(f"[WARNING] info.txt not found for {subject}, skipping...")
        continue

    # Read metadata
    data = {}
    with open(info_file, "r") as f:
        for line in f:
            if ":" in line:
                key, value = [part.strip() for part in line.split(":", 1)]
                data[key.lower()] = value

    # Create participant dict
    participant = {
        "id": subj_id,
        "height": float(data.get("height", 0)),
        "weight": float(data.get("weight", 0)),
        "gender": data.get("gender")
    }

    # Save YAML
    yaml_path = os.path.join(output_folder, f"{subj_id}.yaml")
    with open(yaml_path, "w") as out:
        yaml.dump(participant, out, sort_keys=False)

    print(f"Saved YAML for {subject}: {yaml_path}")
