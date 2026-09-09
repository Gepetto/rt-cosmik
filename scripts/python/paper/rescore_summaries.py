#!/usr/bin/env python3
"""Rewrite the sweep summaries against the mocap modality as the reference.

The dataset's published joint angles come from a different biomechanical model,
so scoring against them measures model mismatch as well as estimation -- about
7 to 8 deg of it, enough to reverse the upper-limb conclusion. They are not used
anywhere in the evaluation.

The reference is the mocap markers driven through the same model and the same
IK as every other modality, which makes the study a clean ablation: one solver,
one model, one marker convention, and only the input changes.

No pipeline is re-run; every modality is already on disk.

    python3 scripts/python/paper/rescore_summaries.py
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import sweep as sweep_mod

CONFIGS = ["mmpose_0-2-4-6", "mmpose_0-2", "nlf_0-2-4-6", "nlf_0-2", "nlf_0"]
REFERENCE_TAG = "mocap_reference"


def main():
    from rtcosmik.config_loader import settings
    ev = sweep_mod._load_eval()
    root = Path(settings.output_dir)
    out_dir = REPO / "results" / "vs_mocap"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in CONFIGS:
        source = REPO / "results" / f"{name}.csv"
        if not source.exists():
            print(f"  {name}: no summary, skipped")
            continue
        rows, kept = list(csv.DictReader(open(source))), []
        for row in rows:
            if row.get("status") != "ok":
                continue
            run_dir = root / row["participant"] / row["task"] / name
            ref_dir = root / row["participant"] / row["task"] / REFERENCE_TAG
            if not (run_dir.is_dir() and ref_dir.is_dir()):
                continue
            try:
                row.update(sweep_mod.score(run_dir, ref_dir, ev))
                kept.append(row)
            except Exception as exc:
                print(f"  {name} {row['participant']}/{row['task']}: "
                      f"{type(exc).__name__}: {exc}")
        target = out_dir / f"{name}.csv"
        with open(target, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sweep_mod.FIELDS)
            writer.writeheader()
            for row in kept:
                writer.writerow({k: row.get(k, "") for k in sweep_mod.FIELDS})
        print(f"  {name}: {len(kept)} trials -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
