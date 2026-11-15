"""``timing.md`` from the passes of sustained_timing.py."""
import csv
import json
from pathlib import Path

import numpy as np

from rtcosmik.paper.timing import missed_deadlines, paced_latency, statistics
from sustained_timing import (CAMERAS, CONFIGURATIONS, FILTER_LAG_FRAMES, IK_TYPES, STAGES,
                              WARMUP_FRAMES)

PASSES = {"replay": "COMFI", "opencap": "OpenCap LabValidation"}


def _trials(folder):
    """{trial: {column: array}} of a pass's per-frame files, warm-up frames left out."""
    trials = {}
    for path in sorted(folder.glob("*.csv")):
        with open(path) as handle:
            rows = list(csv.DictReader(handle))[WARMUP_FRAMES:]
        trials[path.stem] = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    return trials


def _line(label, stats):
    return f"| {label} | {stats['median']:.2f} | {stats['p95']:.2f} | {stats['p99']:.2f} | {stats['max']:.1f} |"


def replay_section(folder):
    """Stage table and latency of every configuration of one pass."""
    lines = []
    for name, (backend, detection) in CONFIGURATIONS.items():
        trials = _trials(folder / name)
        if not trials:
            continue
        described = json.loads((folder / name / "pass.json").read_text())
        period_ms = described["period_ms"]
        pooled = {key: np.concatenate([t[key] for t in trials.values()]) for key in next(iter(trials.values()))}
        frames = len(pooled["processing"])
        lines += [f"### NLF backend `{backend}`, detection `{detection}`", "",
                  f"{len(trials)} trials, {frames} frames, cameras {described['cameras']}, "
                  f"{1e3 / period_ms:.0f} Hz.", "",
                  "| Stage (ms) | Median | 95th | 99th | Max |", "|---|---|---|---|---|"]
        lines += [_line(stage, statistics(pooled[stage])) for stage in STAGES]
        lines.append(_line("processing, decode excluded", statistics(pooled["processing"])))
        with_decode = pooled["processing"] + pooled["decode"]
        lines.append(_line("processing and decode", statistics(with_decode)))
        lines.append("")
        latency = np.concatenate([paced_latency(t["processing"], period_ms) for t in trials.values()])
        replayed = np.concatenate([paced_latency(t["processing"] + t["decode"], period_ms)
                                   for t in trials.values()])
        stats = statistics(latency)
        lag = FILTER_LAG_FRAMES * period_ms
        lines += [
            f"- Processing rate: {1e3 / np.median(pooled['processing']):.0f} frames/s "
            f"({1e3 / np.median(with_decode):.0f} with video decoding).",
            f"- Frames not finished when the next one arrives ({period_ms:.1f} ms): "
            f"{missed_deadlines(latency, period_ms)} of {frames} "
            f"({100 * missed_deadlines(latency, period_ms) / frames:.2f} %); "
            f"with video decoding {missed_deadlines(replayed, period_ms)} "
            f"({100 * missed_deadlines(replayed, period_ms) / frames:.2f} %).",
            f"- Delay from a frame's arrival to its joint angles: median {stats['median']:.1f} ms, "
            f"95th {stats['p95']:.1f}, 99th {stats['p99']:.1f}, max {stats['max']:.1f}.",
            f"- Latency of a motion event, as waiting for the next frame (0 to {period_ms:.1f} ms), "
            f"processing and filter lag ({lag:.0f} ms): median "
            f"{period_ms / 2 + stats['median'] + lag:.0f} ms, at most "
            f"{period_ms + stats['p99'] + lag:.0f} ms for 99 % of frames.", ""]
    return lines


def ik_section(out):
    lines = ["| IK (ms per frame) | Median | 95th | 99th | Max |", "|---|---|---|---|---|"]
    medians = {}
    for ik_type, label in IK_TYPES.items():
        trials = _trials(out / "ik" / ik_type)
        if not trials:
            return []
        pooled = np.concatenate([t["ik"] for t in trials.values()])
        medians[ik_type] = np.median(pooled)
        lines.append(_line(label, statistics(pooled)))
    lines += ["", f"The moving-horizon IK costs {medians['mhe'] - medians['sbs']:+.2f} ms per frame "
              "more than the per-frame IK (medians).", ""]
    return lines


def timestamps_section(out):
    path = out / "timestamps.csv"
    if not path.exists():
        return []
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    value = lambda key: np.array([float(r[key]) for r in rows])
    frames = int(value("frames").sum())
    lines = [f"{len(rows)} trials, {frames} frames per camera.", "",
             "| Camera | Mean interval (ms) | Frames missing against the nominal rate | Intervals over "
             "1.5 periods | Longest interval (ms) | Offset to camera "
             f"{CAMERAS[0]} (ms, range over trials) |", "|---|---|---|---|---|---|"]
    for camera in CAMERAS:
        missing, long = value(f"missing_{camera}"), value(f"long_{camera}")
        offset = value(f"offset_{camera}_ms")
        mean = np.average(value(f"interval_mean_{camera}_ms"), weights=value("frames"))
        lines.append(f"| {camera} | {mean:.2f} | {int(missing.sum())} ({100 * missing.sum() / frames:.2f} %) | "
                     f"{int(long.sum())} ({100 * long.sum() / frames:.2f} %) | "
                     f"{value(f'interval_max_{camera}_ms').max():.0f} | {offset.min():.1f} to {offset.max():.1f} |")
    lines += ["", "Spread of the four cameras' times within a frame: median over trials of the "
              f"per-trial median {np.median(value('spread_median_ms')):.1f} ms, of the 95th percentile "
              f"{np.median(value('spread_p95_ms')):.1f} ms; largest in any frame "
              f"{value('spread_max_ms').max():.0f} ms.", ""]
    return lines


def write(out):
    out = Path(out)
    machine = json.loads((out / "machine.json").read_text()) if (out / "machine.json").exists() else {}
    lines = ["# Sustained timing", "",
             f"Machine: {machine.get('gpu', '?')}, {machine.get('cores', '?')} cores, load average "
             f"{machine.get('load_average', float('nan')):.1f} at start, {machine.get('date', '')}.", "",
             "Rates are processing rates on recorded video, frames released at the camera rate; a live "
             f"system is capped by its cameras. The first {WARMUP_FRAMES} frames of each trial are left "
             "out. Acquisition and synchronisation delays are not in these figures.", ""]
    for folder, title in PASSES.items():
        section = replay_section(out / folder)
        if section:
            lines += [f"## Pipeline stages, {title}", ""] + section
    lines += ["## Inverse kinematics (COMFI)", ""] + ik_section(out)
    lines += ["## Camera timestamps of the COMFI recordings", ""] + timestamps_section(out)
    (out / "timing.md").write_text("\n".join(lines))
