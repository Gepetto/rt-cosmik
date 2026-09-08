#!/usr/bin/env python3
"""Aggregate sweep summaries into the comparison tables for the paper.

Reads one or more CSVs written by sweep.py and reports, per configuration
(arm x camera count), the accuracy over the trials both arms completed.

    python3 scripts/python/paper/report.py results/*.csv

Only trials present for *every* configuration are used for the headline table,
so the configurations are compared on identical data rather than on whatever
each happened to finish. Trials dropped for this reason are listed.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                for key in ("joint_rmse_mean", "joint_rmse_median", "marker_mm",
                            "freeflyer_mm", "fps", "sync_r", "ik_ms_median"):
                    row[key] = float(row[key]) if row.get(key) else np.nan
                rows.append(row)
    return rows


def config_of(row):
    cameras = row["cameras"].split("-")
    return f"{row['arm']}/{len(cameras)}cam"


def table(title, configs, values, unit, width=22):
    print(f"\n{title}")
    header = f"{'':{width}}" + "".join(f"{c:>16}" for c in configs)
    print(header)
    print("-" * len(header))
    for label, per_config in values:
        line = f"{label:{width}.{width}}"
        for config in configs:
            v = per_config.get(config)
            line += f"{v:>16.2f}" if v is not None and np.isfinite(v) else f"{'-':>16}"
        print(line)
    print("-" * len(header) + f"  {unit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--by-task", action="store_true", help="also break down per task")
    args = ap.parse_args()

    rows = load(args.summaries)
    if not rows:
        raise SystemExit("no successful rows in the given summaries")

    by_config = defaultdict(dict)
    for row in rows:
        by_config[config_of(row)][(row["participant"], row["task"])] = row
    configs = sorted(by_config)

    shared = set.intersection(*(set(v) for v in by_config.values()))
    everything = set().union(*(set(v) for v in by_config.values()))
    print(f"configurations: {', '.join(configs)}")
    print(f"trials: {len(everything)} seen, {len(shared)} completed by all "
          f"configurations and used below")
    if everything - shared:
        missing = sorted(everything - shared)
        print(f"  excluded ({len(missing)}): "
              f"{', '.join('/'.join(m) for m in missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}")

    def stat(config, field, function=np.mean, keys=None):
        keys = shared if keys is None else keys
        values = [by_config[config][k][field] for k in keys
                  if k in by_config[config]]
        values = [v for v in values if np.isfinite(v)]
        return function(values) if values else np.nan

    # Each configuration's own coverage, so a configuration that failed trials is
    # visible rather than silently shrinking the shared set for everyone else.
    print("\ncoverage")
    for config in configs:
        own = stat(config, "joint_rmse_mean", keys=set(by_config[config]))
        print(f"  {config:<16} {len(by_config[config]):>4} trials completed, "
              f"own mean {own:.2f} deg")

    table("Accuracy and speed, over the trials every configuration completed",
          configs, [
        ("joint RMSE mean", {c: stat(c, "joint_rmse_mean") for c in configs}),
        ("joint RMSE median", {c: stat(c, "joint_rmse_mean", np.median) for c in configs}),
        ("marker error mm", {c: stat(c, "marker_mm") for c in configs}),
        ("free-flyer mm", {c: stat(c, "freeflyer_mm") for c in configs}),
        ("pipeline fps", {c: stat(c, "fps") for c in configs}),
        ("IK ms median", {c: stat(c, "ik_ms_median") for c in configs}),
    ], "deg / mm / fps / ms")

    if args.by_task:
        tasks = sorted({t for _, t in shared})
        table("Joint RMSE by task", configs,
              [(task, {c: np.mean([by_config[c][k]["joint_rmse_mean"]
                                   for k in shared if k[1] == task])
                       for c in configs}) for task in tasks], "deg", width=22)
    return 0


if __name__ == "__main__":
    sys.exit(main())
