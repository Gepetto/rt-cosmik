#!/usr/bin/env python3
"""
Interpolate robot joint states at camera timestamps (tz-safe).

- Parses/normalizes timestamp columns to tz-aware UTC.
- (Optional) Estimates a constant clock offset (robot - camera) and shifts robot times.
- Interpolates numeric robot columns at camera times with method='time'.
- No extrapolation: values outside robot time span are NaN (or dropped).
- Optional gap masking: invalidate interpolations spanning > max_gap_ms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from typing import Union

PREF_TS_NAMES = ["timestamp", "time", "ts", "date", "datetime", "header.stamp", "stamp"]

# -------------------- Helpers --------------------

def pick_ts_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in PREF_TS_NAMES:
        if name in cols_lower:
            return cols_lower[name]
    for c in df.columns:
        lc = c.lower()
        if ("time" in lc) or ("stamp" in lc) or ("date" in lc):
            return c
    return df.columns[0]

def normalize_to_datetime(s: pd.Series) -> pd.Series:
    """
    Convert to tz-aware pandas datetime (UTC) from strings or numeric epochs (ns/us/ms/s).
    """
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().mean() > 0.9:
        return dt  # already tz-aware UTC

    x = pd.to_numeric(s, errors="coerce").values
    if not np.isfinite(x).any():
        return dt  # whatever we parsed above

    d = np.diff(x[np.isfinite(x)])
    d = d[np.isfinite(d)]
    if len(d) == 0:
        sec = x.astype(float)
        return pd.to_datetime(sec, unit="s", utc=True, errors="coerce")

    med = float(np.median(d[d > 0])) if np.any(d > 0) else float(np.median(np.abs(d)))
    if med > 1e6:
        scale = 1e9   # ns
    elif med > 1e3:
        scale = 1e6   # us
    elif med > 1:
        scale = 1e3   # ms
    else:
        scale = 1.0   # s

    sec = x / scale
    return pd.to_datetime(sec, unit="s", utc=True, errors="coerce")

def load_timeseries(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    ts_col = pick_ts_column(df)
    df["_time"] = normalize_to_datetime(df[ts_col])
    df = (
        df[df["_time"].notna()]
        .sort_values("_time")
        .drop_duplicates(subset=["_time"], keep="last")
        .reset_index(drop=True)
    )
    return df, ts_col

def ensure_utc_index(ts_like: Union[pd.Series, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """
    Return a tz-aware (UTC) DatetimeIndex without dropping tz info.
    """
    if isinstance(ts_like, pd.Series):
        if ts_like.dtype == "datetime64[ns, UTC]":
            return pd.DatetimeIndex(ts_like)
        # If somehow tz-naive leaked in, localize to UTC (assumes UTC semantics)
        if ts_like.dt.tz is None:
            return pd.DatetimeIndex(ts_like.dt.tz_localize("UTC"))
        return pd.DatetimeIndex(ts_like.dt.tz_convert("UTC"))
    elif isinstance(ts_like, pd.DatetimeIndex):
        if ts_like.tz is None:
            return ts_like.tz_localize("UTC")
        return ts_like.tz_convert("UTC")
    else:
        # Fallback
        return pd.DatetimeIndex(pd.to_datetime(ts_like, utc=True))

def estimate_clock_offset_sec(
    cam_time: pd.Series, rob_time: pd.Series, nearest_tol: Optional[pd.Timedelta] = None
) -> float:
    """
    Median nearest estimate of offset (robot - camera) in seconds.
    """
    cam = pd.DataFrame({"_cam_time": ensure_utc_index(cam_time)}).sort_values("_cam_time")
    rob = pd.DataFrame({"_rob_time": ensure_utc_index(rob_time)}).sort_values("_rob_time")

    tol = nearest_tol if nearest_tol is not None else pd.to_timedelta(10, "s")
    merged = pd.merge_asof(
        cam, rob, left_on="_cam_time", right_on="_rob_time", direction="nearest", tolerance=tol
    )
    dt = (merged["_rob_time"] - merged["_cam_time"]).dt.total_seconds().dropna()
    return float(np.median(dt.values)) if len(dt) else 0.0

def union_interpolate_at_targets(rob_df: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate numeric columns of rob_df at target_times using method='time'.
    No extrapolation: values outside robot time span remain NaN.
    """
    num_cols = rob_df.select_dtypes(include=[np.number]).columns.tolist()
    work = rob_df.set_index("_time")[num_cols].sort_index()

    # Keep tz on both indexes
    target_times = ensure_utc_index(target_times)
    union_index = work.index.union(target_times)
    work_u = work.reindex(union_index)
    work_i = work_u.interpolate(method="time", limit_direction="both")

    # Extract only target times
    return work_i.reindex(target_times)

def mask_large_gaps_by_neighbors(
    result: pd.DataFrame, rob_times: pd.DatetimeIndex, max_gap_ms: float
) -> pd.DataFrame:
    """
    Invalidate rows where the larger of (prev/next) robot sample distance exceeds max_gap_ms.
    """
    if result.empty:
        return result

    target_ns = ensure_utc_index(result.index).asi8  # int64 nanoseconds
    rob_ns = ensure_utc_index(rob_times).asi8

    pos = np.searchsorted(rob_ns, target_ns, side="left")
    prev_ok = pos > 0
    next_ok = pos < len(rob_ns)

    prev_idx = np.where(prev_ok, pos - 1, -1)
    next_idx = np.where(next_ok, pos, -1)

    prev_gap_ns = np.where(prev_ok, target_ns - rob_ns[prev_idx], np.inf)
    next_gap_ns = np.where(next_ok, rob_ns[next_idx] - target_ns, np.inf)

    max_gap_ms_arr = np.maximum(prev_gap_ns, next_gap_ns) / 1e6  # ns → ms
    mask = max_gap_ms_arr <= max_gap_ms

    out = result.copy()
    out[~mask] = np.nan
    return out

# -------------------- Main pipeline --------------------

def interpolate_robot_at_camera(
    camera_csv: str,
    robot_csv: str,
    out_csv: Optional[str] = None,
    estimate_offset: bool = False,
    offset_tol_s: float = 10.0,
    drop_outside: bool = True,
    max_gap_ms: Optional[float] = None,
) -> pd.DataFrame:
    """
    Interpolate robot joint states at camera timestamps.

    Args:
        camera_csv: path to camera timestamps CSV.
        robot_csv:  path to robot joint states CSV.
        out_csv:    optional path to save the aligned/interpolated table.
        estimate_offset: True to estimate constant clock offset (robot - camera).
        offset_tol_s: tolerance window used for offset estimation (seconds).
        drop_outside: drop camera times strictly outside robot time span.
        max_gap_ms:  invalidate interpolations spanning more than this (ms).

    Returns:
        DataFrame with columns: _cam_time (UTC), original camera ts col, interpolated robot columns.
    """
    cam_df_raw, cam_ts_col = load_timeseries(Path(camera_csv))
    rob_df_raw, rob_ts_col = load_timeseries(Path(robot_csv))

    cam_df = cam_df_raw.rename(columns={"_time": "_cam_time"})
    rob_df = rob_df_raw.rename(columns={"_time": "_rob_time"})

    # Optional clock offset (robot - camera), shift robot times
    if estimate_offset:
        offset_sec = estimate_clock_offset_sec(
            cam_df["_cam_time"], rob_df["_rob_time"], nearest_tol=pd.to_timedelta(offset_tol_s, "s")
        )
        print(f"[info] Estimated clock offset (robot - camera) = {offset_sec:.6f} s")
        rob_df["_rob_time"] = rob_df["_rob_time"] - pd.to_timedelta(offset_sec, unit="s")

    # Build camera target index, PRESERVING TZ (no .values here)
    cam_times = ensure_utc_index(cam_df["_cam_time"]).unique().sort_values()

    # Drop camera times outside robot span (safer when cameras started earlier)
    rob_min = rob_df["_rob_time"].min()
    rob_max = rob_df["_rob_time"].max()
    if drop_outside:
        cam_times = cam_times[(cam_times >= rob_min) & (cam_times <= rob_max)]

    # Prepare robot frame (numeric columns only)
    rob_numeric_cols = rob_df.select_dtypes(include=[np.number]).columns.tolist()
    if rob_ts_col in rob_numeric_cols:
        rob_numeric_cols.remove(rob_ts_col)

    rob_for_interp = (
        rob_df[["_rob_time"] + rob_numeric_cols]
        .sort_values("_rob_time")
        .drop_duplicates(subset=["_rob_time"], keep="last")
        .rename(columns={"_rob_time": "_time"})
    )

    # Interpolate at camera times
    interp = union_interpolate_at_targets(rob_df=rob_for_interp, target_times=cam_times)

    # Optional gap masking
    if max_gap_ms is not None:
        interp = mask_large_gaps_by_neighbors(
            interp, rob_times=ensure_utc_index(rob_for_interp["_time"]), max_gap_ms=max_gap_ms
        )

    # Assemble output (keep original camera ts column for convenience)
    cam_out = cam_df.set_index("_cam_time").reindex(cam_times)
    out = cam_out[[cam_ts_col]].copy()
    out.index.name = "_cam_time"
    out = out.join(interp)

    if out_csv:
        out.reset_index().to_csv(out_csv, index=False)
        print(f"[ok] Wrote: {out_csv}  (rows: {len(out)})")

    return out.reset_index()


# ---------- Example usage ----------
if __name__ == "__main__":
    SUBJECTS = [
    "Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie","Guilhem",
    "Kahina","Marie_M","Mathis","Maxime_","Mohamed","Nicolas","Zoe","Herbert","Emmanuelle"
]
    TASKS = ["robot_sanding", "robot_welding"]   # <-- put your tasks here

    for subject in SUBJECTS:
        for task in TASKS:
            CAMERA_CSV = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/mouv/{task}/camera_0_timestamps.csv"
            ROBOT_CSV  = f"/root/workspace/ros_ws/src/rt-cosmik/output/robot/robot_data_csv/{subject}_{task}.csv"
            OUTPUT_CSV = f"/root/workspace/ros_ws/src/rt-cosmik/output/robot/aligned_data/{subject}_{task}.csv"

            try:
                df_out = interpolate_robot_at_camera(
                    camera_csv=CAMERA_CSV,
                    robot_csv=ROBOT_CSV,
                    out_csv=OUTPUT_CSV,
                    estimate_offset=False,   # set True if you suspect a constant clock offset
                    offset_tol_s=10.0,
                    drop_outside=True,       # important since cameras started earlier
                    max_gap_ms=100.0         # None to disable gap masking
                )
                print(f"[ok] Done: {subject} - {task} ({len(df_out)} rows)")
            except Exception as e:
                print(f"[ERROR] {subject} - {task}: {e}")
