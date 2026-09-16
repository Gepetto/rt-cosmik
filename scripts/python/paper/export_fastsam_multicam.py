#!/usr/bin/env python3
"""Export FastSAM markers for every camera into one flat folder per trial.

Produces, for each participant and task::

    <out>/<participant_id>/<Task>/cosmik_mhr_markers_cam{0,2,4,6}.csv

Cameras 2, 4 and 6 are extracted from the raw FastSAM results with the
colleague's own exporter (``export_cosmik_mhr_markers_csv.py``, imported rather
than copied), using the same marker map as camera 0. So the four files come out
of one code path and one set of vertex indices; only the camera differs.

Camera 0 was exported earlier and COMFI ships it as
``cosmik_mhr_markers_cam.csv``; it is copied across under the indexed name.

COMFI is mounted read-only in the container, so the tree is written to a
staging folder and copied into ``COMFI/fastsam`` on the host.

Participant 3361 is left out of cameras 2/4/6: its FastSAM results come from a
different inference script and do not line up with COMFI's calibration. Its
camera-0 file is still renamed, so the folder layout is uniform.

    python3 scripts/python/paper/export_fastsam_multicam.py \\
        --results /root/workspace/COMFI/fastsam/results_multicam \\
        --comfi-fastsam /root/workspace/COMFI/fastsam \\
        --out <staging folder, then copied into COMFI/fastsam>
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

CAMERAS = (2, 4, 6)
EXCLUDED_IDS = {"3361"}
LEGACY_CAM0 = "cosmik_mhr_markers_cam.csv"


def csv_name(camera):
    return f"cosmik_mhr_markers_cam{camera}.csv"


def load_exporter(results_root):
    path = results_root / "export_cosmik_mhr_markers_csv.py"
    spec = importlib.util.spec_from_file_location("fastsam_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject_id(mapping, folder):
    """Resolve a results folder to a COMFI id, tolerating the Maxime/Maxime_ spelling."""
    ids = mapping["subject_ids"]
    for candidate in (folder, folder.rstrip("_"), folder + "_"):
        if candidate in ids:
            return ids[candidate]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--comfi-fastsam", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    exporter = load_exporter(args.results)
    mapping = json.loads(exporter.DEFAULT_MAP.read_text())
    body_names, body_idx, face_names, face_idx = exporter.ordered_definitions(mapping)
    names = body_names + face_names
    vertex_count = int(mapping["vertex_count"])

    exported, copied, skipped = 0, 0, []
    for folder in sorted(p for p in args.results.iterdir() if p.is_dir()):
        pid = subject_id(mapping, folder.name)
        if pid is None:
            skipped.append(f"{folder.name}: no id in the marker map")
            continue
        if args.participants and pid not in args.participants:
            continue
        for raw_task, task in mapping["task_directories"].items():
            target = args.out / pid / task
            target.mkdir(parents=True, exist_ok=True)

            legacy = args.comfi_fastsam / pid / task / LEGACY_CAM0
            dest0 = target / csv_name(0)
            if legacy.exists() and (args.overwrite or not dest0.exists()):
                shutil.copyfile(legacy, dest0)
                copied += 1

            if pid in EXCLUDED_IDS:
                continue
            for camera in CAMERAS:
                result_dir = folder / raw_task / f"camera_{camera}" / "calibrated"
                output = target / csv_name(camera)
                if output.exists() and not args.overwrite:
                    continue
                if not (result_dir / "vertices_cam.npy").exists():
                    skipped.append(f"{pid}/{task}/cam{camera}: no vertices_cam.npy")
                    continue
                loaded = exporter.load_standard(result_dir, vertex_count, body_idx, face_idx)
                points, frames, persons, valid = exporter.validate_points(
                    *loaded, len(names), result_dir)
                exporter.write_csv(output, names, points, frames, persons, valid,
                                   overwrite=True)
                exported += 1
                print(f"{pid}/{task}/cam{camera}: {len(points)} rows, "
                      f"{int((~valid).sum())} invalid", flush=True)

    # Participants that only exist as camera-0 files (3361) still get renamed.
    for legacy in sorted(args.comfi_fastsam.glob(f"*/*/{LEGACY_CAM0}")):
        pid, task = legacy.parts[-3], legacy.parts[-2]
        if not pid.isdigit() or (args.participants and pid not in args.participants):
            continue
        dest0 = args.out / pid / task / csv_name(0)
        if not dest0.exists():
            dest0.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy, dest0)
            copied += 1

    print(f"\nexported {exported} camera files, copied {copied} camera-0 files")
    for item in skipped:
        print(f"  skipped {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
