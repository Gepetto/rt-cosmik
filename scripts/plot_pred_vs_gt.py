#!/usr/bin/env python3
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- marker sets (must match what you trained / logged) ----------
UPPER_MKS = [
    'r_lelbow_study','r_melbow_study','r_lwrist_study','r_mwrist_study',
    'L_lelbow_study','L_melbow_study','L_lwrist_study','L_mwrist_study'
]
LOWER_MKS = [
    'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
    'r_knee_study','r_mknee_study','r_ankle_study','r_mankle_study',
    'r_toe_study','r_5meta_study','r_calc_study',
    'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
    'L_toe_study','L_calc_study','L_5meta_study',
    'r_shoulder_study','L_shoulder_study','C7_study'
]
AXES = ['x','y','z']

def get_mks(body_part):
    if body_part == 'upper':
        return UPPER_MKS
    if body_part == 'lower':
        return LOWER_MKS
    raise ValueError("body-part must be upper|lower")

def idx_of(marker, axis, mks):
    """Return flat column index for GT_/Pred_ format: 3*marker_idx + axis_idx."""
    j = mks.index(marker)
    a = AXES.index(axis)
    return 3*j + a

def find_named_columns(df, marker, axis):
    gt_col = f"GT.{marker}.{axis}"
    pr_col = f"Pred.{marker}.{axis}"
    if gt_col in df.columns and pr_col in df.columns:
        return gt_col, pr_col
    # Also accept 'GT_marker_axis' style
    gt_alt = f"GT_{marker}_{axis}"
    pr_alt = f"Pred_{marker}_{axis}"
    if gt_alt in df.columns and pr_alt in df.columns:
        return gt_alt, pr_alt
    return None, None

def maybe_denormalize(series, height):
    if height is None:
        return series
    return series * height

def main():
    ap = argparse.ArgumentParser(description="Plot GT vs Pred for one marker component over time.")
    ap.add_argument("--csv", required=True, help="Path to wide CSV logged by PredictionLogger")
    ap.add_argument("--body-part", choices=["upper","lower"], required=True)
    ap.add_argument("--marker", required=True, help="Marker name (must match training order)")
    ap.add_argument("--axis", choices=["x","y","z"], default="x")
    ap.add_argument("--height", type=float, default=None, help="If provided, de-normalize by multiplying (→ meters)")
    ap.add_argument("--out", type=str, default=None, help="Optional path to save the figure (PNG).")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    mks = get_mks(args.body_part)

    # Try “named” columns first
    gt_col, pr_col = find_named_columns(df, args.marker, args.axis)
    if gt_col is None:
        # Fallback to indexed GT_i / Pred_i layout
        # Expect header: Frame, GT_0, Pred_0, GT_1, Pred_1, ...
        base = idx_of(args.marker, args.axis, mks)
        gt_col = f"GT_{base}"
        pr_col = f"Pred_{base}"
        for c in (gt_col, pr_col):
            if c not in df.columns:
                raise KeyError(
                    f"Column '{c}' not found.\n"
                    "If your CSV has named columns (e.g., GT.marker.axis), pass matching --marker/--axis.\n"
                    "If it is indexed GT_i/Pred_i, ensure body-part/marker order matches your training."
                )

    t = df["Frame"] if "Frame" in df.columns else np.arange(len(df), dtype=int)
    gt = maybe_denormalize(df[gt_col].to_numpy(), args.height)
    pr = maybe_denormalize(df[pr_col].to_numpy(), args.height)

    # Plot 2D time-series
    plt.figure(figsize=(10,5))
    plt.plot(t, gt, label="GT")
    plt.plot(t, pr, label="Pred", linestyle="--")
    ttl = args.title or f"{args.marker}·{args.axis}   ({'meters' if args.height else 'normalized units'})"
    plt.title(ttl)
    plt.xlabel("Frame")
    plt.ylabel("Position" + ( " [m]" if args.height else " [norm]" ))
    plt.legend()
    plt.grid(True, alpha=0.3)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
