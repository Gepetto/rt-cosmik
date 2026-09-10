#!/usr/bin/env python3
"""Where to put FastSAM's missing ``Head`` marker, and whether it matters.

FastSAM exports 34 of the 35 parity markers plus two thoracic markers parity
drops. The one it does not export is ``Head``, which cannot simply be omitted:
``construct_segments_frames`` only builds the head segment when Head, REar and
LEar are all present, and ``get_head_pose`` takes the head's vertical axis from
``Head - shoulder_centre``. Without it this arm would solve a structurally
different model from every other arm.

Two questions, two modes:

``measure`` -- where do the other modalities put Head, relative to the facial
landmarks FastSAM does export? Reported in a head-local frame built from the
ears and the nose and scaled by ear width, so it is comparable across subjects.
This is where :data:`rtcosmik.paper.fastsam_source.HEAD_OFFSET` comes from.

``sensitivity`` -- re-solve trials with deliberately different placements. If
the spread across plausible offsets is small next to the gap between arms, the
choice is not what the comparison rests on.

    python3 scripts/python/paper/study_head_offset.py measure
    python3 scripts/python/paper/study_head_offset.py sensitivity
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

TASKS = ["Screwing", "Polishing", "SideOverhead", "RobotPolishing",
         "RobotWelding", "Lifting"]
MEASURE_ARMS = ["nlf_0", "nlf_0-2", "nlf_0-2-4-6", "mmpose_0-2-4-6"]

#: Trials for the sensitivity pass: six participants, six different tasks.
SENSITIVITY_TRIALS = [("1012", "Lifting"), ("1118", "Screwing"),
                      ("4279", "SideOverhead"), ("1602", "RobotPolishing"),
                      ("2112", "Polishing"), ("4801", "RobotWelding")]

#: Alternative placements, in units of ear width: (forward, up, lateral).
VARIANTS = {
    "measured (nlf_0)": (-0.216, 0.881, -0.058),
    "nlf 4-cam offset": (-0.203, 0.858, -0.029),
    "straight up only": (0.0, 0.881, 0.0),
    "10% shorter head": (-0.216, 0.793, -0.058),
    "20 deg forward": (0.085, 0.874, -0.058),
}


def read_markers(path, names):
    header = open(path).readline().strip().split(",")
    if any(f"{n}_x" not in header for n in names):
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None]
    return {n: data[:, [header.index(f"{n}_{a}") for a in "xyz"]] for n in names}


def head_in_face_frame(markers):
    """Head relative to the ear midpoint, in ear widths: (forward, up, lateral).

    The same construction :func:`fastsam_source.derive_head` inverts.
    """
    midpoint = (markers["REar"] + markers["LEar"]) / 2
    lateral = markers["REar"] - markers["LEar"]
    width = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral = lateral / width
    forward = markers["Nose"] - midpoint
    forward = forward - (forward * lateral).sum(1, keepdims=True) * lateral
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)
    up = np.cross(lateral, forward)
    d = markers["Head"] - midpoint
    local = np.stack([(d * forward).sum(1), (d * up).sum(1),
                      (d * lateral).sum(1)], 1) / width
    return local, width[:, 0]


def measure():
    from rtcosmik.config_loader import settings

    root = Path(settings.output_dir)
    participants = sorted(p.name for p in root.iterdir() if p.is_dir())
    print("Head in the ear/nose frame, in units of ear width, "
          "mean +/- std over trials.\n")
    print(f"{'arm':<18}{'forward':>16}{'up':>16}{'lateral':>16}"
          f"{'ear width mm':>14}{'trials':>8}")
    for arm in MEASURE_ARMS:
        rows, widths = [], []
        for participant in participants:
            for task in TASKS:
                path = root / participant / task / arm / "markers.csv"
                if not path.exists():
                    continue
                markers = read_markers(path, ["REar", "LEar", "Nose", "Head"])
                if markers is None:
                    continue
                local, width = head_in_face_frame(markers)
                ok = np.isfinite(local).all(1)
                if ok.sum() > 10:
                    rows.append(local[ok].mean(0))
                    widths.append(width[ok].mean())
        if not rows:
            print(f"{arm:<18}{'-- no runs --':>16}")
            continue
        a = np.asarray(rows)
        line = f"{arm:<18}"
        for k in range(3):
            line += f"{a[:, k].mean():>10.3f}+-{a[:, k].std():<4.3f}"
        print(line + f"{np.mean(widths) * 1000:>14.1f}{len(rows):>8}")
    print("\nMoCap is absent by construction: it carries a Vicon head band, "
          "not facial\nlandmarks, so it has no Nose to build the frame from -- "
          "and it is the\nreference, so deriving from it would tune the estimate "
          "to its own truth.")


def sensitivity(dataset, work_dir):
    import sweep as sweep_mod
    from rtcosmik.config_loader import settings
    from rtcosmik.paper import fastsam_source

    ev = sweep_mod._load_eval()
    work = Path(work_dir)
    results = {}
    for label, offset in VARIANTS.items():
        fastsam_source.HEAD_OFFSET = offset
        scores = []
        for participant, task in SENSITIVITY_TRIALS:
            out = work / label.replace(" ", "_").replace("%", "pc")
            out = out / f"{participant}_{task}"
            try:
                sweep_mod.run_fastsam(dataset, participant, task, [0], out, settings)
                scores.append(sweep_mod.score(
                    out, Path(dataset) / "mocap" / "aligned" / participant / task,
                    ev)["joint_rmse_mean"])
            except Exception as exc:
                print(f"  {label} {participant}/{task}: {type(exc).__name__}: {exc}")
                scores.append(np.nan)
        results[label] = np.asarray(scores, dtype=float)
        print(f"{label:<20} mean {np.nanmean(results[label]):6.3f} deg   "
              f"{[round(s, 2) for s in scores]}", flush=True)

    base = results["measured (nlf_0)"]
    print("\nChange in whole-body joint RMSE against the measured placement:\n")
    print(f"{'placement':<20}{'mean':>10}{'worst trial':>14}")
    for label, scores in results.items():
        d = scores - base
        print(f"{label:<20}{np.nanmean(d):>+10.3f}{np.nanmax(np.abs(d)):>14.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure", "sensitivity"])
    ap.add_argument("--dataset", default="/root/workspace/COMFI")
    ap.add_argument("--work-dir", default=str(REPO / "results" / "head_offset"),
                    help="Where the sensitivity pass writes its throwaway runs")
    args = ap.parse_args()
    if args.mode == "measure":
        return measure() or 0
    return sensitivity(args.dataset, args.work_dir)


if __name__ == "__main__":
    sys.exit(main())
