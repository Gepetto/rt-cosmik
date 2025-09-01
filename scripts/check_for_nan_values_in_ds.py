#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

dataset_path = Path("/home/ngouget/Codes/datasets/COSMIK_dataset_mixed")

def has_no_nan(csv_path: str) -> bool:
    """
    Returns False if the CSV contains any NaN (missing) values,
    True otherwise.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return False

    return not df.isnull().values.any()

def main():
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")

    for subject in sorted(p.name for p in dataset_path.iterdir() if p.is_dir()):
        subject_path = dataset_path / subject

        for entry in sorted(subject_path.iterdir()):
            # Trials should be directories; skip stray files
            if not entry.is_dir():
                continue

            trial = entry.name
            trial_path = subject_path / trial

            # Build expected CSV paths
            jcp_csv      = trial_path / f"{trial}_jcp_hpe.csv"
            mks_csv      = trial_path / f"{trial}_mks_rt.csv"
            # devices_csv  = trial_path / f"{trial}_devices.csv"

            # Check for NaN values in CSVs
            has_nan_jcp = has_no_nan(jcp_csv)
            has_nan_mks = has_no_nan(mks_csv)
            # has_nan_dev = has_no_nan(devices_csv)

            if not has_nan_jcp or not has_nan_mks:
                print(f"[WARN] Found NaN values in {subject} {trial}")
                continue


            # If there are other non-CSV assets in trial dir you want to keep:
            # for f in trial_path.iterdir():
            #     if f.is_file() and f.suffix.lower() not in {".csv"}:
            #         shutil.copy2(f, out_trial / f.name)

if __name__ == "__main__":
    main()


