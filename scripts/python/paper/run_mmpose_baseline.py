#!/usr/bin/env python3
"""Run the old mmpose + OpenCap-LSTM front end through the current IK.

Produces joint_angles.csv and markers.csv in the ordinary layout, so
``eval/compare_to_mocap.py`` reads a baseline run and an NLF run side by side
with no special casing.

Requires ``marker_set = "parity"`` in settings.py: the baseline cannot produce
the other 8 markers, and running it against the full 43-marker model would
compare two different models rather than two pose estimators. The same setting
must be used for the NLF arm it is compared against.

    python3 scripts/python/paper/run_mmpose_baseline.py \
        --dataset COMFI --participant 1012 --task Lifting
"""
import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import yaml

from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from rtcosmik.filtering.iir import IIR
from rtcosmik.paper.mmpose_baseline import HALPE26, MmposeMarkerSource, load_trial
from rtcosmik.pipeline.solver import HumanSolver
from rtcosmik.saver.csv_saver import CSVSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True)
LOGGER = logging.getLogger("mmpose_baseline")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset root, COMFI layout")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cameras", type=int, nargs="+", default=list(settings.cameras),
                    help="Camera ids; the first is the triangulation reference")
    ap.add_argument("--out", default=None, help="Output directory")
    args = ap.parse_args()

    if settings.marker_set != "parity":
        raise SystemExit(
            f"settings.marker_set is {settings.marker_set!r}; the mmpose baseline "
            f"needs 'parity'. It cannot produce T11, T6 or the hand markers, and "
            f"comparing it against the full model would compare two models "
            f"rather than two pose estimators.")

    root = Path(args.dataset)
    out_dir = Path(args.out) if args.out else (
        Path(settings.output_dir) / args.participant / args.task
        / f"{len(args.cameras)}cam_mmpose_lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = yaml.safe_load((root / "metadata" / f"{args.participant}.yaml").read_text())
    LOGGER.info(f"subject {args.participant}: {meta['height']} m, {meta['weight']} kg, "
                f"{meta['gender']}")

    cam_dir = root / "cam_params" / args.participant
    mtxs, dists, projections, _, _ = load_camera_parameters(cam_dir, args.cameras)
    world_R, world_T = load_world_transformation(cam_dir, args.cameras[0])

    mmpose_dir = root / "mmpose" / "output" / args.participant / args.task
    keypoints, confidences = load_trial(mmpose_dir, args.task, args.cameras)
    LOGGER.info(f"{keypoints.shape[0]} frames x {keypoints.shape[1]} cameras "
                f"from {mmpose_dir}")

    from rtcosmik.augmenter.marker_augmenter import loadModel
    augmenter_dir = SRC_ROOT / "rtcosmik" / "augmenter" / "augmentation_model"
    models = loadModel(augmenterDir=str(augmenter_dir), augmenterModelName="LSTM",
                       augmenter_model="v0.3")
    LOGGER.info(f"loaded OpenCap v0.3 LSTM augmenter from {augmenter_dir}")

    iir = IIR(num_channel=3 * len(HALPE26), sampling_frequency=settings.fs)
    iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                   filter_type=settings.filter_type)

    source = MmposeMarkerSource(
        keypoints, confidences, mtxs, dists, projections, world_R, world_T,
        models, augmenter_dir, meta["height"], meta["weight"],
        iir=iir, buffer_len=settings.N, logger=LOGGER)

    solver = HumanSolver(settings, gender=meta["gender"][0],
                         height=meta["height"], weight=meta["weight"],
                         logger=LOGGER)
    csv = CSVSaver(str(out_dir),
                   markers_header=["Frame_0"] + list(settings.marker_names),
                   # No frame column here: run_pipeline writes joint angles as
                   # bare joint names, and compare_to_mocap matches on them.
                   joint_angles_header=list(settings.joint_angles_names))

    ik_ms, calib_ms, rows = [], None, 0
    started = time.perf_counter()
    for frame, mks in source:
        t0 = time.perf_counter()
        q = solver.solve(mks)
        elapsed = (time.perf_counter() - t0) * 1e3
        if calib_ms is None:
            # The first call builds and calibrates the model; keeping it in the
            # per-frame statistics would misreport the steady-state cost.
            calib_ms = elapsed
        else:
            ik_ms.append(elapsed)

        # Flattened per axis, exactly as run_pipeline writes it, so the two
        # arms produce byte-comparable CSVs for the evaluation script.
        markers = OrderedDict([("Frame_0", frame)])
        for name in settings.marker_names:
            position = mks[name]
            markers[f"{name}_x"] = float(position[0])
            markers[f"{name}_y"] = float(position[1])
            markers[f"{name}_z"] = float(position[2])
        angles = OrderedDict(
            zip(settings.joint_angles_names, (float(v) for v in q)))
        csv.save_markers(markers)
        csv.save_joint_angles(angles)
        rows += 1
        if rows % 200 == 0:
            LOGGER.info(f"{rows}/{len(source)} frames, "
                        f"{rows/(time.perf_counter()-started):.1f} fps")
    csv.close()

    ik = np.asarray(ik_ms)
    LOGGER.info(f"wrote {rows} rows to {out_dir}")
    LOGGER.info(f"model calibration {calib_ms:.0f} ms (excluded from the IK stats)")
    if len(ik):
        LOGGER.info(f"IK per frame: mean {ik.mean():.2f} ms, median "
                    f"{np.median(ik):.2f}, p95 {np.percentile(ik,95):.2f}, "
                    f"max {ik.max():.2f}")
    LOGGER.info(f"end to end {rows/(time.perf_counter()-started):.1f} fps "
                f"(triangulation + LSTM + IK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
