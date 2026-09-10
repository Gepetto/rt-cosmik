#!/usr/bin/env python3
"""How rigid is each modality's skeleton?

A real body's bones do not change length. Every arm's ``markers.csv`` holds the
landmarks that were actually fed to the IK, so measuring how much a segment's
length wanders over a trial says whether that arm's landmarks move as a linked
body or as independent points.

The distinction is the whole argument for the 3D arms. Triangulation solves each
landmark separately and imposes nothing between them, so its segment lengths
breathe with the noise on each endpoint. A network that regresses a metric body
pose per view carries the linkage in its own model, and fusing views preserves
it. This script puts a number on that, per arm.

Two numbers per arm, because rigidity alone proves nothing:

*Wander* -- the within-trial standard deviation of segment length, averaged over
trials. Not the spread of the mean across participants, which is mostly genuine
differences in build. Low wander means the landmarks move as a linked body.

*Bias* -- how far the arm's mean segment length sits from mocap's, on the same
trial. A parametric body model can hold a bone perfectly steady at entirely the
wrong length, which would look excellent in the wander column and still put the
joint centres in the wrong place. Reading the first column without the second is
the mistake this script exists to prevent.

    python3 scripts/python/paper/study_coherence.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

TASKS = ["Screwing", "Polishing", "SideOverhead", "RobotPolishing",
         "RobotWelding", "Lifting"]
ARMS = [
    ("mmpose_0-2", "mmpose 2 cams"),
    ("mmpose_0-2-4-6", "mmpose 4 cams"),
    ("nlf2d_0-2", "NLF-2D tri 2 cams"),
    ("nlf2d_0-2-4-6", "NLF-2D tri 4 cams"),
    ("nlf_0", "NLF-3D 1 cam"),
    ("nlf_0-2", "NLF-3D 2 cams"),
    ("nlf_0-2-4-6", "NLF-3D 4 cams"),
    ("fastsam_0", "FastSAM-3D 1 cam"),
    ("mocap_reference", "MoCap"),
]
SEGMENTS = [
    ("upper arm R", "RSHO", "RELB"),
    ("forearm R", "RELB", "RWRI"),
    ("upper arm L", "LSHO", "LELB"),
    ("forearm L", "LELB", "LWRI"),
    ("shank R", "RKNE", "RANK"),
    ("shank L", "LKNE", "LANK"),
    ("pelvis width", "RASI", "LASI"),
    ("shoulders", "RSHO", "LSHO"),
]


def segment_lengths(path):
    """Per-frame length of every segment in one run, in mm."""
    needed = sorted({m for _, a, b in SEGMENTS for m in (a, b)})
    columns = [f"{m}_{axis}" for m in needed for axis in "xyz"]
    try:
        frame = pd.read_csv(path, usecols=lambda c: c in columns)
    except Exception:
        return None
    if not all(c in frame.columns for c in columns):
        return None
    point = {m: frame[[f"{m}_{a}" for a in "xyz"]].to_numpy(float) for m in needed}
    out = {}
    for label, a, b in SEGMENTS:
        d = np.linalg.norm(point[a] - point[b], axis=1) * 1000
        d = d[np.isfinite(d)]
        if len(d) > 10:
            out[label] = (float(d.mean()), float(d.std()))
    return out


def main():
    from rtcosmik.config_loader import settings

    root = Path(settings.output_dir)
    participants = sorted(p.name for p in root.iterdir() if p.is_dir())

    # Mocap first: every other arm's bias is measured against it, trial by trial,
    # so that differences in build cancel instead of adding to the number.
    truth = {}
    for participant in participants:
        for task in TASKS:
            path = root / participant / task / "mocap_reference" / "markers.csv"
            if path.exists():
                lengths = segment_lengths(path)
                if lengths:
                    truth[(participant, task)] = lengths

    wander, bias, counts = {}, {}, {}
    for tag, label in ARMS:
        w = {s[0]: [] for s in SEGMENTS}
        b = {s[0]: [] for s in SEGMENTS}
        trials = 0
        for participant in participants:
            for task in TASKS:
                path = root / participant / task / tag / "markers.csv"
                if not path.exists():
                    continue
                lengths = segment_lengths(path)
                if not lengths:
                    continue
                trials += 1
                reference = truth.get((participant, task), {})
                for name, (mean, std) in lengths.items():
                    w[name].append(std)
                    if name in reference:
                        b[name].append(mean - reference[name][0])
        if trials:
            wander[label] = {n: (np.mean(v) if v else np.nan) for n, v in w.items()}
            bias[label] = {n: (np.mean(v) if v else np.nan) for n, v in b.items()}
            counts[label] = trials

    if not wander:
        raise SystemExit("no runs found under output/")

    names = [s[0] for s in SEGMENTS]
    width = max(len(l) for l in wander) + 2

    print("Within-trial segment-length wander (std over the trial, mm), averaged")
    print("over trials. Lower means the landmarks move as a linked body.\n")
    print(" " * width + "".join(f"{n[:12]:>13}" for n in names)
          + f"{'mean':>9}{'n':>6}")
    for label, row in wander.items():
        values = [row[n] for n in names]
        print(f"{label:<{width}}" + "".join(f"{v:>13.1f}" for v in values)
              + f"{np.nanmean(values):>9.1f}{counts[label]:>6}")

    print("\n\nSegment-length bias against mocap on the same trial (mm). Positive")
    print("means the arm's bones are longer than the markers say they should be.")
    print("A rigid skeleton of the wrong size scores well above and badly here.\n")
    print(" " * width + "".join(f"{n[:12]:>13}" for n in names)
          + f"{'mean|.|':>9}")
    for label, row in bias.items():
        if label == "MoCap":
            continue
        values = [row[n] for n in names]
        print(f"{label:<{width}}" + "".join(f"{v:>+13.1f}" for v in values)
              + f"{np.nanmean(np.abs(values)):>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
