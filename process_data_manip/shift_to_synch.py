import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt
from src.rtcosmik.utils.read_write_utils import read_joint_angles_wholebody
from scipy.signal import correlation_lags

def synchronize_signals(sig1, sig2):
    """
    Synchronize two signals by shifting sig2 relative to sig1.

    Args:
        sig1: numpy array, reference signal
        sig2: numpy array, signal to be shifted

    Returns:
        lag: number of samples sig2 was shifted (+ means sig2 delayed)
    """

    corr = correlate(sig1, sig2, mode="full")
    lags = correlation_lags(len(sig1), len(sig2), mode="full")
    lag = lags[np.argmax(corr)]
    return lag

def apply_shift_to_dataframe(df, lag):
    """
    Shift all columns of dataframe by lag.
    Positive lag = delay df, Negative lag = advance df
    """
    if lag > 0:
        pad = pd.DataFrame(np.nan, index=range(lag), columns=df.columns)
        df_shifted = pd.concat([pad, df], ignore_index=True).iloc[:len(df)]
    elif lag < 0:
        df_shifted = df.iloc[-lag:].reset_index(drop=True)
        pad = pd.DataFrame(np.nan, index=range(-lag), columns=df.columns)
        df_shifted = pd.concat([df_shifted, pad], ignore_index=True)
    else:
        df_shifted = df.copy()
    return df_shifted

if __name__ == "__main__":
    no_trial = "Anais"
    task = "jump"
    path_mocap  = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/q_mocap.csv"
    path_cosmik = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/cosmik_2cams/{task}/q_cosmik_ipopt.csv"

    # Load all joint angles into DataFrames
    df_cosmik = pd.read_csv(path_cosmik)
    df_mocap  = pd.read_csv(path_mocap)

    # Use knee angle to compute lag
    knee_cosmik = df_cosmik["Rknee_flex_ext"].values
    knee_mocap  = df_mocap["Rknee_flex_ext"].values

    lag = synchronize_signals(knee_cosmik, knee_mocap)
    print(f"Best lag (samples): {lag}")

    # Apply lag to mocap data
    df_mocap_aligned = apply_shift_to_dataframe(df_mocap, lag)

    # Save aligned mocap
    save_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/mocap/{task}/q_mocap_aligned.csv"
    df_mocap_aligned.to_csv(save_path, index=False)
    print(f"Aligned mocap saved to: {save_path}")

    # --- Optional: visualize only the knee for verification ---
    for col in ["Rhip_flex_ext", "Rknee_flex_ext", "Rankle_flex_ext"]:
        plt.figure()
        plt.plot(df_cosmik[col].values, label=f"Cosmik {col}")
        plt.plot(df_mocap[col].values, label=f"Mocap {col} (orig)")
        plt.plot(df_mocap_aligned[col].values, "--", label=f"Mocap {col} (aligned)")
        plt.legend()
        plt.title(col)
        plt.show()