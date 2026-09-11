#!/usr/bin/env python3
"""Per-stage cost of the SMPL-refined NLF pipeline, in the pipeline.

Component timings measured in isolation do not add up to what the pipeline
achieves: NLF, YOLO and the fitter share one GPU, so each is slower with the
others running than it is alone. This runs the real loop and times each stage
inside it, which is the only number a deployment decision can be made on.

Reports, per configuration: NLF inference, the SMPL fit, the IK, and the
end-to-end rate over the same frames.

    python3 scripts/python/paper/bench_nlfsmpl.py
    python3 scripts/python/paper/bench_nlfsmpl.py --frames 400
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

#: (label, cameras, fit?, online iters, warm start)
CONFIGS = [
    ("NLF only",                    [0],          False, 0, False),
    ("NLF + SMPL, 1 it, warm",      [0],          True,  1, True),
    ("NLF + SMPL, 1 it, cold",      [0],          True,  1, False),
    ("NLF + SMPL, 4 it, warm",      [0],          True,  4, True),
    ("NLF only",                    [0, 2, 4, 6], False, 0, False),
    ("NLF + SMPL, 1 it, warm",      [0, 2, 4, 6], True,  1, True),
    ("NLF + SMPL, 1 it, cold",      [0, 2, 4, 6], True,  1, False),
    ("NLF + SMPL, 4 it, warm",      [0, 2, 4, 6], True,  4, True),
]


def run(dataset, participant, task, cameras, use_fit, num_iter, warm, frames,
        settings):
    import torch
    import yaml
    from rtcosmik.camera.cam_utils import (load_camera_parameters,
                                           load_world_transformation)
    from rtcosmik.filtering.iir import IIR
    from rtcosmik.nlf.nlf import extract_views
    from rtcosmik.pipeline.solver import HumanSolver
    from rtcosmik.triangulation.triangulation import reconstruct_3d
    from rtcosmik.utils.VideoReader import OfflineVideoSource
    import sweep as sweep_mod

    root = Path(dataset)
    meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
    cam_dir = root / "cam_params" / participant
    mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, cameras)
    world_R, world_T = load_world_transformation(cam_dir, cameras[0])

    # The fitting arm needs the dense cloud; the plain arm needs only markers.
    indices = list(range(settings.smpl_num_vertices)) if use_fit else None
    estimator = sweep_mod._nlf_estimator(mtxs, len(cameras), settings,
                                         indices=indices)
    refiner = None
    if use_fit:
        from rtcosmik.smpl.fitter import SmplRefiner
        refiner = SmplRefiner(
            gender=meta["gender"][0], num_betas=settings.smpl_num_betas,
            num_iter=num_iter, beta_mode="calibrated",
            calibration_frames=settings.smpl_calibration_frames,
            calibration_iter=settings.smpl_calibration_iter, warm_start=warm,
            model_root=settings.body_models_path, device=settings.device,
            compile_online=settings.smpl_compile)
        refiner.warmup()

    solver = HumanSolver(settings, gender=meta["gender"][0], height=meta["height"],
                         weight=meta["weight"])
    names = list(settings.marker_names)
    rows = np.asarray(settings.nlf_indices)
    iir = IIR(num_channel=3 * len(names), sampling_frequency=settings.fs)
    iir.add_filter(order=settings.order, cutoff=settings.cutoff_freq,
                   filter_type=settings.filter_type)

    source = OfflineVideoSource(
        paths=[str(root / "videos" / participant / task / f"camera_{c}.mp4")
               for c in cameras],
        size_wh=(settings.width, settings.height), loop=False)

    from collections import deque
    buffer = deque(maxlen=settings.N)
    nlf_ms, fit_ms, ik_ms, total_ms = [], [], [], []
    seen = 0
    while seen < frames:
        images = source.read()
        if images is None:
            break
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        out, _, _, _ = estimator.estimate_from_frames(images)
        views = extract_views(out, len(cameras))
        p3d = reconstruct_3d(views, projections)
        t_nlf = (time.perf_counter() - t0) * 1e3
        if len(p3d) == 0:
            continue

        t_fit = 0.0
        if refiner is not None:
            t0 = time.perf_counter()
            p3d = refiner.refine(np.asarray(p3d))
            t_fit = (time.perf_counter() - t0) * 1e3
            p3d = p3d[rows]

        p3d = np.asarray(p3d) @ np.asarray(world_R).T + np.asarray(world_T)
        if not buffer:
            for _ in range(settings.N):
                buffer.append(p3d)
        else:
            buffer.append(p3d)
        block = np.asarray(buffer).reshape(settings.N, 3 * len(names))
        xyz = iir.filter(block).reshape(settings.N, len(names), 3)[-1]

        t0 = time.perf_counter()
        solver.solve(dict(zip(names, xyz)))
        t_ik = (time.perf_counter() - t0) * 1e3

        seen += 1
        if seen > settings.smpl_calibration_frames + 10:   # past warm-up
            nlf_ms.append(t_nlf)
            fit_ms.append(t_fit)
            ik_ms.append(t_ik)
            total_ms.append((time.perf_counter() - t_start) * 1e3)
    source.release()          # or ffmpeg lives on holding GPU memory
    del estimator, refiner
    torch.cuda.empty_cache()
    return (np.median(nlf_ms), np.median(fit_ms), np.median(ik_ms),
            np.median(total_ms), len(total_ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/root/workspace/COMFI")
    ap.add_argument("--participant", default="1012")
    ap.add_argument("--task", default="Lifting")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    from rtcosmik.config_loader import settings

    print(f"{args.participant}/{args.task}, median ms per frame, measured inside "
          f"the running pipeline\n")
    print(f"{'cams':>5}  {'configuration':<26}{'NLF':>8}{'SMPL':>8}{'IK':>8}"
          f"{'total':>9}{'fps':>8}{'n':>6}")
    for label, cams, use_fit, iters, warm in CONFIGS:
        try:
            nlf, fit, ik, total, n = run(
                args.dataset, args.participant, args.task, cams, use_fit, iters,
                warm, args.frames, settings)
            print(f"{len(cams):>5}  {label:<26}{nlf:>8.2f}{fit:>8.2f}{ik:>8.2f}"
                  f"{total:>9.2f}{1000 / total:>8.1f}{n:>6}", flush=True)
        except Exception as exc:
            print(f"{len(cams):>5}  {label:<26}FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
    print("\nSMPL column is the fit only. 'warm' starts each pose solve from the")
    print("previous frame; 'cold' starts from rest. Shape is calibrated once over")
    print(f"the first {settings.smpl_calibration_frames} frames at "
          f"{settings.smpl_calibration_iter} iterations in every fitted row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
