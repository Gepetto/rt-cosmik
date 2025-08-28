
#!/usr/bin/env python3
"""
Sync 1000 Hz force data (no timestamps) to irregular camera timestamps (~40 Hz).

Assumptions:
- The first row of the force CSV corresponds to the FIRST timestamp in the camera CSV.
- Force CSV has columns "Frame" and "Sub Frame" (mocap at 100 Hz, subframe 0..9), followed by numeric channels.
- Force sampling rate is 1000 Hz (configurable via --force-fs).
- Camera CSV has two columns: frame_index, timestamp (timestamp as "%Y-%m-%d %H:%M:%S.%f"), but may also be headerless.

Output:
- A CSV aligned to the camera timestamps with columns:
    ["camera_frame", "timestamp", <interpolated force channels...>]
- Optionally includes nearest "Frame" and "Sub Frame" from the force file if present.
"""

import argparse
import sys
import pandas as pd
import numpy as np

def read_camera_timestamps(path: str) -> pd.DataFrame:
    """
    Read camera timestamps with robust header handling.
    Returns a DataFrame with columns ["camera_frame", "timestamp"] where timestamp is pd.Timestamp.
    """
    # First try: no header provided
    raw = pd.read_csv(path, header=None)
    # Detect if first row contains headers
    if raw.shape[1] >= 2 and str(raw.iloc[0,0]).strip().lower() in {"frame_index", "frame", "index"}:
        # Read again with header row
        df = pd.read_csv(path)
        # Normalize column names
        cols = [c.strip().lower() for c in df.columns]
        # Try to find frame and timestamp columns
        if "frame_index" in cols:
            fi = cols.index("frame_index")
        elif "frame" in cols:
            fi = cols.index("frame")
        else:
            # Fallback to first column
            fi = 0
        if "timestamp" in cols:
            ti = cols.index("timestamp")
        else:
            # Fallback to second column
            ti = 1
        cam = pd.DataFrame({
            "camera_frame": df.iloc[:, fi].astype(int).to_numpy(),
            "timestamp": pd.to_datetime(df.iloc[:, ti].astype(str), format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
        })
    else:
        # Assume first column is frame index, second is timestamp string
        cam = pd.DataFrame({
            "camera_frame": raw.iloc[:,0].astype(int).to_numpy(),
            "timestamp": pd.to_datetime(raw.iloc[:,1].astype(str), format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
        })
    if cam["timestamp"].isna().any():
        raise ValueError("Failed to parse some camera timestamps.")
    # Ensure strictly increasing timestamps (required by interpolation)
    cam = cam.sort_values("timestamp").reset_index(drop=True)
    return cam

def linear_time_interp(force_times_ns: np.ndarray, Y: np.ndarray, query_times_ns: np.ndarray) -> np.ndarray:
    """
    Vectorized linear interpolation for multiple columns.
    - force_times_ns: shape (N,), int64 nanoseconds (monotonic ascending)
    - Y: shape (N, C), float64 (NaNs allowed; will be linearly filled before interp)
    - query_times_ns: shape (M,), int64 nanoseconds (monotonic ascending)

    Returns: shape (M, C) array of interpolated values (clipped to endpoints).
    """
    # Fill NaNs in Y along time with linear interpolation (per column)
    Y = pd.DataFrame(Y).interpolate(method="linear", axis=0, limit_direction="both").to_numpy(dtype=float)

    # For each column, use numpy.interp (1D). Loop over columns for simplicity/clarity.
    out = np.empty((query_times_ns.shape[0], Y.shape[1]), dtype=float)
    x = force_times_ns.astype(np.float64)
    xp = query_times_ns.astype(np.float64)
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(float)
        out[:, j] = np.interp(xp, x, y)  # clips outside the range to boundary values
    return out

def sync_forces_to_camera(force_csv: str, camera_csv: str, out_csv: str, force_fs: float = 1000.0) -> None:
    # Read inputs
    forces = pd.read_csv(force_csv)
    cam = read_camera_timestamps(camera_csv)

    if forces.shape[0] < 2:
        raise ValueError("Force CSV has fewer than 2 rows; cannot interpolate.")
    if cam.shape[0] < 1:
        raise ValueError("Camera CSV is empty.")

    # Identify non-numeric columns and numeric channels to interpolate
    # Keep 'Frame' and 'Sub Frame' separately if they exist.
    cols = list(forces.columns)
    has_frame = "Frame" in cols
    has_subframe = "Sub Frame" in cols or "SubFrame" in cols

    frame_col = "Frame" if has_frame else None
    subframe_col = "Sub Frame" if "Sub Frame" in cols else ("SubFrame" if "SubFrame" in cols else None)

    numeric_cols = []
    for c in cols:
        if c in {frame_col, subframe_col}:
            continue
        # Consider as numeric if pandas says so after coercion
        try:
            pd.to_numeric(forces[c], errors="raise")
            numeric_cols.append(c)
        except Exception:
            # skip strictly non-numeric (e.g., strings)
            pass

    if len(numeric_cols) == 0:
        raise ValueError("No numeric force channels found for interpolation.")

    # Build force time axis as datetimes starting from FIRST camera timestamp (t0)
    t0 = cam["timestamp"].iloc[0]
    N = forces.shape[0]
    # datetime64[ns] array for force samples
    force_times = (t0.to_datetime64() + (np.arange(N) * (1e9/force_fs)).astype("timedelta64[ns]"))
    # Ensure strictly increasing
    if not np.all(np.diff(force_times.astype("int64")) > 0):
        raise AssertionError("Generated force time axis is not strictly increasing.")

    # Prepare for interpolation
    force_times_ns = force_times.astype("int64")  # ns since epoch
    cam_times_ns = cam["timestamp"].astype("int64").to_numpy()

    # Interpolate numeric channels at camera timestamps
    Y = forces[numeric_cols].to_numpy(dtype=float)
    Y_cam = linear_time_interp(force_times_ns, Y, cam_times_ns)

    # Prepare output
    out = pd.DataFrame(Y_cam, columns=numeric_cols)
    out.insert(0, "timestamp", cam["timestamp"].values)
    out.insert(0, "camera_frame", cam["camera_frame"].values)

    # Optionally append nearest Frame/SubFrame from force file
    if frame_col is not None or subframe_col is not None:
        # find nearest force index for each camera timestamp
        idx = np.searchsorted(force_times_ns, cam_times_ns)
        idx = np.clip(idx, 0, N-1)
        # choose closer neighbor between idx and idx-1
        left = np.clip(idx-1, 0, N-1)
        choose_left = (np.abs(force_times_ns[left] - cam_times_ns) <= np.abs(force_times_ns[idx] - cam_times_ns))
        nearest = np.where(choose_left, left, idx)
        if frame_col is not None:
            out.insert(2, "force_Frame_nearest", forces.iloc[nearest][frame_col].to_numpy())
        if subframe_col is not None:
            out.insert(3 if frame_col is not None else 2, "force_SubFrame_nearest", forces.iloc[nearest][subframe_col].to_numpy())

    # Save
    out.to_csv(out_csv, index=False)
    # Print a short summary
    print(f"Synced CSV written to: {out_csv}")
    print(f"Rows (camera frames): {out.shape[0]} | Channels: {len(numeric_cols)}")
    print("First 5 timestamps:")
    print(out['timestamp'].head().to_string(index=False))

def main():
    force_csv = '/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/squat/squat_devices.csv'
    camera_csv = '/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mouv/squat/camera_0_timestamps.csv'
    out_csv = '/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/squat/force_resampled_40Hz.csv'
    force_fs = 1000

    sync_forces_to_camera(force_csv, camera_csv, out_csv, force_fs)

if __name__ == "__main__":
    main()
