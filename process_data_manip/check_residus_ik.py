import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np

def load_csv_to_array(csv_path, mks_to_skip=None):
    """
    Load CSV with marker data into numpy array of shape (N, J, 3).
    Drops 'Frame' column if present and skips unwanted markers.
    """
    df = pd.read_csv(csv_path)

    # Drop Frame if exists
    if "Frame" in df.columns:
        df = df.drop(columns=["Frame"])

    # Drop unwanted markers
    if mks_to_skip is not None:
        drop_cols = []
        for m in mks_to_skip:
            drop_cols.extend([c for c in df.columns if c.startswith(m + "_")])
        df = df.drop(columns=drop_cols)

    # Build array
    cols = df.columns
    J = len(cols) // 3
    N = len(df)
    
    data = np.zeros((N, J, 3))
    joint_names = []
    
    for j in range(J):
        base = cols[j*3:(j+1)*3]
        joint_names.append(base[0].rsplit("_", 1)[0])  # marker name
        data[:, j, :] = df[base].to_numpy()

    return data, joint_names

def plot_ref_est_ft_concatenated(
    ref,          # (N, J, 3)  in REF/MOCAP order
    est,          # (N, J, 3)  already reordered to REF order
    ft,           # (N, J, 3)  already reordered to REF order
    mapping=None, # IGNORED for indexing; only used for optional names
    ft_in_ref_order=True,   # IGNORED
    title=None
):
    """
    Reordering is REMOVED. Assumes ref, est, ft are already aligned: shape (N, J, 3), same J and order.
    'mapping' is only used (optionally) to display joint names; it does not affect indexing.
    """

    N, J, _ = ref.shape
    # Optional names (from mapping keys) purely for labels; fallback to J0..J{J-1}
    if isinstance(mapping, dict) and len(mapping) > 0:
        joint_names = list(mapping.keys())
        if len(joint_names) != J:
            joint_names = [joint_names[j] if j < len(joint_names) else f"J{j}" for j in range(J)]
    else:
        joint_names = [f"J{j}" for j in range(J)]

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    offset = 0
    boundaries = []
    name_positions = []
    rmse_points_est = []
    rmse_points_ft  = []

    for j in range(J):
        ref_j = ref[:, j, :]/1000  # (N,3)
        est_j = est[:, j, :]
        ft_j  = ft[:,  j, :]

        t = np.arange(N) + offset

        # Plot x,y,z
        for d, ax in enumerate(axes[:3]):
            ax.plot(t, ref_j[:, d], "k-", linewidth=1, label="mks_" if (offset == 0 and d == 0) else None)
            ax.plot(t, est_j[:, d], "r-", linewidth=1, label="mks_model" if (offset == 0 and d == 0) else None)
            # ax.plot(t, ft_j[:,  d], "g-", linewidth=1, label="fine-tuned" if (offset == 0 and d == 0) else None)
            if d == 0:
                name_positions.append((offset + N/2, joint_names[j]))

        # Framewise residual norms
        res_est = np.linalg.norm(est_j - ref_j, axis=1)
        # res_ft  = np.linalg.norm(ft_j  - ref_j, axis=1)

        # Per-joint RMSE (scalar over all coords & frames)
        rmse_est = np.sqrt(np.mean((est_j - ref_j)**2))
        # rmse_ft  = np.sqrt(np.mean((ft_j  - ref_j)**2))

        axes[3].plot(t, res_est, color="r", linewidth=0.6)
        # axes[3].plot(t, res_ft,  color="g", linewidth=0.6, alpha=0.35)

        center = offset + N/2
        rmse_points_est.append((center, rmse_est, joint_names[j]))
        # rmse_points_ft.append((center, rmse_ft,  joint_names[j]))

        offset += N
        boundaries.append(offset)

    # Vertical boundaries
    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)

    # Joint name labels on top subplot
    top_ax = axes[0]
    y_top = top_ax.get_ylim()[1]
    for x, name in name_positions:
        top_ax.text(x, y_top, name, ha="center", va="top", fontsize=8, rotation=90)

    # Labels/legend
    for d, ax in enumerate(axes[:3]):
        ax.set_ylabel(["x","y","z"][d])
        if d == 0:
            ax.legend(loc="upper left", ncol=3, frameon=False)

    axes[3].set_ylabel("Residual / RMSE")
    axes[3].set_xlabel("Frames concatenated per marker")
    axes[3].legend(["ik_residual"], loc="upper left", frameon=False)

    plt.tight_layout()
    plt.show()


mks_to_skip = [
    'LForearm','LUArm','RUArm','RHJC_study','LHJC_study','r_pelvis','l_pelvis',
    'LHL2','LHM5','RHL2','RHM5','LHand','RForearm','RHand',
    'L_sh1_study','L_thigh1_study','r_sh1_study','r_thigh1_study'
]

ref_csv = "/root/workspace/ros_ws/src/rt-cosmik/output/4279/mocap/robot_welding/mocap_downsampled_to_40hz.csv"     # CSV without Frame
est_csv = "/root/workspace/ros_ws/src/rt-cosmik/output/4279/mocap/robot_welding/mks_model_mocap_downsampled.csv"     # CSV with Frame

ref, joint_names = load_csv_to_array(ref_csv,mks_to_skip)
est, _ = load_csv_to_array(est_csv,mks_to_skip)

print("ref shape:", ref.shape)   # (N, J, 3)
print("est shape:", est.shape)   # (N, J2, 3)

# For testing: let’s say ft = est + small noise
ft = est + 0.01*np.random.randn(*est.shape)

# Create mapping {name: index}
mapping = {name: i for i, name in enumerate(joint_names)}

plot_ref_est_ft_concatenated(
    ref=ref,
    est=est,
    ft=ft,
    mapping=mapping,
    title="Residuals: Ref vs Est vs Fine-tuned"
)