import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
no_trial= "Mathis"
task = "static"
df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mouv/{task}/mocap_downsampled_to_40hz.csv")  # replace with your actual file path
# Remove the 'Mathis:' prefix from all column names
df.columns = [col.replace("Mathis:", "") for col in df.columns]
frames = df["Frame"] if "Frame" in df.columns else range(len(df))

# Extract base marker names
marker_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))

for marker in marker_names:
    try:
        print(marker)
        x = df[f"{marker}_x"]
        y = df[f"{marker}_y"]
        z = df[f"{marker}_z"]
    except KeyError:
        continue  # skip if any coordinate is missing
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Trajectory of marker: {marker}")

    axs[0].plot(frames, x, color='r')
    axs[0].set_ylabel("X (mm)")

    axs[1].plot(frames, y, color='g')
    axs[1].set_ylabel("Y (mm)")

    axs[2].plot(frames, z, color='b')
    axs[2].set_ylabel("Z (mm)")
    axs[2].set_xlabel("Frame")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
