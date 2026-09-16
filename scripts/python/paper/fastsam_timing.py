#!/usr/bin/env python3
"""FastSAM-3D inference time per view, from the export logs.

FastSAM runs offline, one camera at a time, on the cluster that produced
``COMFI/fastsam/results_multicam``. Each sequence's ``batch_extract.log`` prints
the wall time of every image as ``[process_one_image] TOTAL: <s>`` -- person
detection, both decoders and the MHR post-processing. The first image of a
sequence also pays for model warm-up and is dropped.

Writes ``timing/fastsam_per_view.csv``: one row per camera sequence, with the
mean, SD, median and 95th percentile of the per-image time in ms. Cameras 2, 4
and 6 only: camera 0 was exported earlier by a different script with no such
log, and all four cameras run the same model on the same image size.

    python3 scripts/python/paper/fastsam_timing.py --out results/campaign/paper
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

import numpy as np

RESULTS = Path("/root/workspace/COMFI/fastsam/results_multicam")
MAP = "cosmik_mhr_marker_map_17subjects_tv8_tv12.json"
TOTAL = re.compile(rb"\[process_one_image\] TOTAL: ([0-9.]+)s")
FIELDS = ["participant", "task", "camera", "frames", "mean_ms", "sd_ms", "median_ms", "p95_ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "paper")
    args = ap.parse_args()

    mapping = json.loads((RESULTS / MAP).read_text())
    ids, tasks = mapping["subject_ids"], mapping["task_directories"]
    rows = []
    for log in sorted(RESULTS.glob("*/*/camera_*/calibrated/batch_extract.log")):
        folder, task, camera = log.parts[-5], log.parts[-4], log.parts[-3]
        participant = next((ids[c] for c in (folder, folder.rstrip("_"), folder + "_") if c in ids),
                           "3361" if folder == "Mathis" else folder)
        seconds = np.array([float(x) for x in TOTAL.findall(log.read_bytes())])[1:]
        if seconds.size == 0:
            continue
        ms = 1000 * seconds
        rows.append({"participant": participant, "task": tasks.get(task, task),
                     "camera": int(camera.split("_")[1]), "frames": int(ms.size),
                     "mean_ms": float(ms.mean()), "sd_ms": float(ms.std(ddof=1)),
                     "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95))})

    target = args.out / "timing" / "fastsam_per_view.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    means = np.array([r["mean_ms"] for r in rows])
    print(f"{len(rows)} sequences, {sum(r['frames'] for r in rows)} images: per view "
          f"{means.mean():.0f} (SD {means.std(ddof=1):.0f}) ms, "
          f"{1000 / means.mean():.1f} images/s -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
