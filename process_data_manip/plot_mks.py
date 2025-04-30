import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
df = pd.read_csv("/root/workspace/ros_ws/src/rt-cosmik/output/trial3/overhead/overhead_clean_trajectories.csv")  # replace with your actual file path
frames = df["Frame"] if "Frame" in df.columns else range(len(df))

# Extract base marker names
marker_names = sorted(set(col.rsplit("_", 1)[0] for col in df.columns if "_x" in col))

# Create a directory to save plots (optional)
os.makedirs("marker_plots", exist_ok=True)

for marker in marker_names:
    try:
        x = df[f"{marker}_x (mm)"]
        y = df[f"{marker}_y (mm)"]
        z = df[f"{marker}_z (mm)"]
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
    plt.close()  # Close to avoid memory issues
