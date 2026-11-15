#!/usr/bin/env python3
"""Height of the head top above the mocap head band, per fold.

For each participant: the median, over a trial, of the NLF head-top landmark's
height above the band's centroid along the band's normal, averaged over trials.
Each fold takes the mean over its fitting participants.

    python3 scripts/python/paper/fit_head_raise.py --runs /root/workspace/oof/runs27 \\
        --arm nlf_0-2-4-6 --dataset /root/workspace/COMFI
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from rtcosmik.folds import PARTICIPANTS
from rtcosmik.paper.mocap_reference import HEAD_BAND, MM_TO_M, band_up, fold_raises


def height(run_dir, mocap_dir):
    """Median height (m) of the run's Head above the band's centroid, along the band's normal."""
    run = pd.read_csv(run_dir / "markers.csv")
    mocap = pd.read_csv(mocap_dir / "markers_trajectories.csv")
    frames = run["Frame_0"].to_numpy().astype(int)
    keep = frames < len(mocap)
    band = {name: mocap[[f"{name}_{axis}[mm]" for axis in "XYZ"]].to_numpy()[frames[keep]] * MM_TO_M
            for name in HEAD_BAND}
    head = run[["Head_x", "Head_y", "Head_z"]].to_numpy()[keep]
    above = ((head - np.mean(list(band.values()), axis=0)) * band_up(band)).sum(1)
    return float(np.nanmedian(above))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    heights = {}
    for participant in PARTICIPANTS:
        trials = sorted((args.runs / participant).glob(f"*/{args.arm}"))
        if not trials:
            raise FileNotFoundError(f"no {args.arm} run for {participant} in {args.runs}")
        heights[participant] = np.mean([height(trial, args.dataset / "mocap" / "aligned" / participant
                                               / trial.parent.name) for trial in trials])
        print(f"{participant}: {1e3 * heights[participant]:.0f} mm over {len(trials)} trials")
    print("HEAD_RAISE_M = (" + ", ".join(f"{value:.4f}" for value in fold_raises(heights)) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
