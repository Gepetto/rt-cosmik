#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

dataset_path = Path("/home/ngouget/Codes/datasets/COSMIK_dataset")
output_path  = Path("/home/ngouget/Codes/datasets/COSMIK_dataset_npy")

def save_csv_as_npy(csv_path: Path, out_npy_path: Path, float_dtype=np.float32):
    if not csv_path.exists():
        print(f"[warn] missing file: {csv_path}")
        return
    arr = pd.read_csv(csv_path).to_numpy(dtype=float_dtype, copy=False)
    np.save(out_npy_path, arr)

def main():
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    for subject in sorted(p.name for p in dataset_path.iterdir() if p.is_dir()):
        subject_path = dataset_path / subject
        out_subject  = output_path  / subject
        out_subject.mkdir(parents=True, exist_ok=True)

        for entry in sorted(subject_path.iterdir()):
            # Trials should be directories; skip stray files
            if not entry.is_dir():
                # If you really want to copy non-dir files into subject root:
                shutil.copy2(entry, out_subject / entry.name)
                continue

            trial = entry.name
            trial_path = subject_path / trial
            out_trial  = out_subject  / trial
            out_trial.mkdir(parents=True, exist_ok=True)

            # Build expected CSV paths
            jcp_csv      = trial_path / f"{trial}_jcp_mocap.csv"
            mks_csv      = trial_path / f"{trial}_trajectories.csv"
            devices_csv  = trial_path / f"{trial}_devices.csv"

            # Convert each CSV → NPZ (compressed) preserving headers
            save_csv_as_npy(jcp_csv,     out_trial / f"{trial}_jcp_mocap.npy")
            save_csv_as_npy(mks_csv,     out_trial / f"{trial}_trajectories.npy")
            save_csv_as_npy(devices_csv, out_trial / f"{trial}_devices.npy")

            # If there are other non-CSV assets in trial dir you want to keep:
            # for f in trial_path.iterdir():
            #     if f.is_file() and f.suffix.lower() not in {".csv"}:
            #         shutil.copy2(f, out_trial / f.name)

if __name__ == "__main__":
    main()


