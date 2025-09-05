#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import sys

dataset_path = Path(sys.argv[1])
output_path  = Path(sys.argv[2])

def save_csv_as_npz(csv_path: Path, out_npz_path: Path, float_dtype=np.float32):
    """Load CSV -> save as .npz with both data and column names."""
    if not csv_path.exists():
        print(f"[warn] missing file: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    # Use .to_numpy for speed and consistent dtype
    data = df.to_numpy(dtype=float_dtype, copy=False)
    cols = df.columns.to_numpy()
    np.savez_compressed(out_npz_path, data=data, columns=cols)

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
            jcp_hpe      = trial_path / f"{trial}_jcp_hpe.csv"
            mks_csv      = trial_path / f"{trial}_mks_rt.csv"

            # Convert each CSV → NPZ (compressed) preserving headers
            save_csv_as_npz(jcp_csv,     out_trial / f"{trial}_jcp_mocap_rt.npz")
            save_csv_as_npz(jcp_hpe,     out_trial / f"{trial}_jcp_hpe.npz")
            save_csv_as_npz(mks_csv,     out_trial / f"{trial}_mks_mocap_rt.npz")

            # If there are other non-CSV assets in trial dir you want to keep:
            # for f in trial_path.iterdir():
            #     if f.is_file() and f.suffix.lower() not in {".csv"}:
            #         shutil.copy2(f, out_trial / f.name)

if __name__ == "__main__":
    main()


