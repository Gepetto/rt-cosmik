#!/usr/bin/env python3
"""
Profile the NLFEstimator.estimate_from_frames() pipeline on provided video inputs.
Forces identical CPU-pinning to the production pipeline, binding execution 
strictly to the designated gpu_worker core subset.
"""
import argparse
import ctypes
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
LOGGER = logging.getLogger(__name__)

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik.nlf.nlf import NLFEstimator
from rtcosmik.utils.videoReader import OfflineVideoSource
from rtcosmik.camera.cam_utils import load_camera_parameters
from rtcosmik.config_loader import settings

width = settings.width
height = settings.height


# ---------------------------------------------------------------------------
# Core-pinning helpers, identical to run_pipeline.py's -- kept in sync so
# this benchmark's gpu_worker core subset matches production exactly rather
# than approximating it.
# ---------------------------------------------------------------------------

def _get_allowed_cpus():
    """The actual CPU IDs this process may run on, respecting cgroup/cpuset
    limits (Docker --cpus, Kubernetes resources.limits.cpu, taskset, etc).
    This is authoritative -- unlike lscpu/nproc, which can show host-wide
    topology even inside a restricted container."""
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return list(range(os.cpu_count() or 1))


def _get_physical_core_id(cpu_id):
    """Physical core ID for a logical CPU, to detect hyperthread siblings."""
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id"
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return cpu_id  # fallback: treat as its own isolated core


def _get_cpu_max_freq_khz(cpu_id):
    """Max frequency this logical CPU can reach -- used to distinguish
    P-cores (high) from E-cores (lower) on hybrid Intel chips, since
    /topology/core_id alone doesn't tell you which is which."""
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq"
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0  # unknown -> treated as lowest priority


def _drop_hyperthread_siblings(cpu_ids):
    """Keep only ONE logical CPU per physical core, dropping the rest.

    Makes this process behave as if hyperthreading were off, WITHOUT
    touching the host's actual SMT state (no /sys writes, no root needed,
    no effect on other processes/containers sharing the machine).

    Real host-level SMT off (if you want the effect to apply system-wide
    instead) is a separate, one-line terminal action:
        echo off > /sys/devices/system/cpu/smt/control   # requires --privileged
        echo on  > /sys/devices/system/cpu/smt/control   # to re-enable
    """
    seen_cores = set()
    kept = []
    for cpu in cpu_ids:
        core = _get_physical_core_id(cpu)
        if core in seen_cores:
            continue
        seen_cores.add(core)
        kept.append(cpu)
    return kept


def _compute_core_pins(allowed_cpus):
    """Same split as run_pipeline.py: rank physical cores by max frequency
    (fastest first, so P-cores are preferred over E-cores on hybrid chips),
    then hand ik_worker the fastest core exclusively, reader the slowest,
    and everything in between to gpu_worker (the subset this script cares
    about, since it's profiling the NLF/YOLO stage specifically)."""
    by_core = {}
    for cpu in allowed_cpus:
        core = _get_physical_core_id(cpu)
        by_core.setdefault(core, []).append(cpu)

    def core_max_freq(cpu_list):
        return max(_get_cpu_max_freq_khz(c) for c in cpu_list)

    physical_cores = sorted(by_core.values(), key=core_max_freq, reverse=True)

    if len(physical_cores) >= 3:
        ik_cpus = physical_cores[0]
        reader_cpus = physical_cores[-1]
        gpu_cpus = [c for core in physical_cores[1:-1] for c in core]
    elif len(physical_cores) == 2:
        ik_cpus = physical_cores[0]
        reader_cpus = physical_cores[1]
        gpu_cpus = physical_cores[1]
    else:
        n = len(allowed_cpus)
        third = max(1, n // 3)
        reader_cpus = allowed_cpus[:third] or allowed_cpus[:1]
        gpu_cpus = allowed_cpus[third:2 * third] or allowed_cpus[:1]
        ik_cpus = allowed_cpus[2 * third:] or allowed_cpus[-1:]

    return {'reader': reader_cpus, 'gpu_worker': gpu_cpus, 'ik_worker': ik_cpus}


def _pin_current_thread_to_cores(core_ids):
    """Pin the calling OS thread/process to a specific set of CPU cores
    (Linux only). This script is single-threaded, so pinning "the current
    thread" pins the whole process."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        cpu_set_t_size = 128
        mask = (ctypes.c_uint8 * cpu_set_t_size)()
        for core_id in core_ids:
            mask[core_id // 8] |= (1 << (core_id % 8))
        SYS_gettid = 186  # x86_64
        tid = libc.syscall(SYS_gettid)
        ret = libc.sched_setaffinity(tid, cpu_set_t_size, ctypes.byref(mask))
        if ret != 0:
            errno = ctypes.get_errno()
            LOGGER.warning(f"[WARN] sched_setaffinity failed for cores {core_ids}: errno={errno}")
        else:
            LOGGER.info(f"[INFO] Pinned process (tid={tid}) to cores {core_ids}")
    except Exception as exc:
        LOGGER.warning(f"[WARN] Could not set CPU affinity to {core_ids}: {exc}")


def run_case(video_paths, width, height, warmup_iters, num_iters, core_pins):
    num_cameras = len(video_paths)
    mtxs, dists, projections, rotations, translations = load_camera_parameters(settings.cam_calib_path, num_cameras)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10, warmup=0, active=35, repeat=11),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )

    prof.start()

    # Bind this process to the same gpu_worker core subset the production
    # pipeline's gpu_worker thread would use, so this profile's numbers are
    # measured under matching CPU-placement conditions -- not just "however
    # many cores happen to be idle right now."
    if core_pins.get('gpu_worker'):
        _pin_current_thread_to_cores(core_pins['gpu_worker'])

    print(f"\n{'='*20} Running Profile ({num_cameras} Input Stream(s)) {'='*20}")
    print(f"Target Video(s): {video_paths}")

    src = OfflineVideoSource(paths=[Path(p) for p in video_paths], size_wh=(width, height))

    est = NLFEstimator(
        yolo_path=settings.yolo_path,
        nlf_path=settings.nlf_path,
        cano_path=settings.cano_path,
        image_size=(width, height),
        cam_Ks=mtxs[:num_cameras],
        indices=settings.nlf_indices,
        conf=settings.yolo_conf,
        imgsz=settings.yolo_imgsz,
        device=settings.device,
        warmup=True,
        warmup_iters=warmup_iters,
    )

    stages = {"yolo_ms": [], "box_select_ms": [], "h2d+pre_ms": [],
              "nlf_ms": [], "post_ms": [], "total_ms": []}

    try:
        for i in range(num_iters):
            frames = src.read()
            if frames is None:
                print(f"[WARN] video source exhausted/stalled at iter {i}, stopping early")
                break

            out, timings, yres, boxes = est.estimate_from_frames(frames)

            t_post0 = time.perf_counter()
            poses2d_list = []
            poses2d = out.get("poses2d") if isinstance(out, dict) else None
            if poses2d is not None:
                for ii in range(num_cameras):
                    p2d = poses2d[ii] if ii < len(poses2d) else None
                    if p2d is not None and len(p2d) > 0 and p2d[0] is not None:
                        poses2d_list.append(p2d[0].detach().float().cpu().numpy())
            torch.cuda.synchronize()
            post_ms = (time.perf_counter() - t_post0) * 1000.0

            stages["yolo_ms"].append(timings["yolo_ms"])
            stages["box_select_ms"].append(timings.get("box_select_ms", 0.0))
            stages["h2d+pre_ms"].append(timings["h2d+pre_ms"])
            stages["nlf_ms"].append(timings["nlf_ms"])
            stages["post_ms"].append(post_ms)
            stages["total_ms"].append(timings["total_ms"] + post_ms)
            prof.step()
    finally:
        src.release()
        prof.stop()
        prof.export_chrome_trace("trace.json")

    print(f"\n{'─'*24} METRICS BREAKDOWN {'─'*24}")
    for name, vals in stages.items():
        if vals:
            print(f"{name:16s}: mean {statistics.mean(vals):7.2f} ms | "
                  f"median {statistics.median(vals):7.2f} ms | "
                  f"max {max(vals):7.2f} ms")
    print(f"{'─'*65}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=str, nargs="+", required=True,
                    help="Video file paths to profile.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--iters", type=int, default=350, help="Timed iterations")
    p.add_argument("--warmup", type=int, default=15, help="Warmup iterations")
    p.add_argument("--no-ht", action="store_true",
                    help="Treat hyperthread siblings as unavailable for core pinning "
                         "(container-scoped, no host/root changes)")
    args = p.parse_args()

    # Match production's torch CPU-threading constraints (run_pipeline.py's
    # main()) so this profile reflects the same contention conditions.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    allowed_cpus = _get_allowed_cpus()
    if args.no_ht:
        before = allowed_cpus
        allowed_cpus = _drop_hyperthread_siblings(allowed_cpus)
        LOGGER.info(f"[INFO] Hyperthreading disabled for this run: {before} -> {allowed_cpus}")

    core_pins = _compute_core_pins(allowed_cpus)
    LOGGER.info(f"[INFO] Process allowed CPUs: {allowed_cpus}")
    LOGGER.info(f"[INFO] Computed core pins: {core_pins}")

    run_case(args.videos, width, height, args.warmup, args.iters, core_pins)


if __name__ == "__main__":
    main()