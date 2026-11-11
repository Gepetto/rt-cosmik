#!/usr/bin/env python3
"""Measure each trial's clock offset between video and reference, once.

COMFI's own comparison script shifts every trial by a knee-flexion lag. Here the
shift is measured on the fused landmarks of the recorded views, before any
filter or IK, so it holds the dataset's offset and none of the pipeline's lag.
Scoring removes it from the causal error of every arm of that trial.

    python3 scripts/python/paper/measure_clock_offsets.py --views /root/workspace/oof/views
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

DATASET = Path("/root/workspace/COMFI")


def measure(views_path, dataset, participant, task):
    """Frames the recorded video trails the reference by, in landmark space."""
    from rtcosmik.config_loader import settings
    from rtcosmik.paper.nlf_views import replay_markers
    from rtcosmik.paper.scoring import reference_dir, unfiltered_marker_lag
    frames = replay_markers(views_path, Path(dataset) / "cam_params" / participant)
    return unfiltered_marker_lag(frames, settings.marker_names,
                                 reference_dir(dataset, participant, task))


def main():
    from rtcosmik.paper.scoring import CLOCK_OFFSETS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True, help="the sweep's --views-cache folder")
    ap.add_argument("--out", type=Path, default=CLOCK_OFFSETS)
    args = ap.parse_args()

    offsets = {}
    for path in sorted(args.views.glob("*/*.npz")):
        participant, task = path.parent.name, path.stem
        offsets.setdefault(participant, {})[task] = measure(path, DATASET, participant, task)
        print(f"{participant}/{task}: {offsets[participant][task]} frames", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(offsets, indent=2) + "\n")
    values = [v for tasks in offsets.values() for v in tasks.values()]
    print(f"{len(values)} trials, offsets {sorted(set(values))} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
