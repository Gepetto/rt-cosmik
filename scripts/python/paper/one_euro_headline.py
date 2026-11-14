#!/usr/bin/env python3
"""Choose the 1-euro parameters per fold and apply the headline filter rule (ticket 09).

Reads the study tables of ``ik_filter_studies.py`` and writes ``one_euro.csv`` (the
cross-validated arm, one row per trial) and ``one_euro_choice.json`` next to them.

    python3 scripts/python/paper/one_euro_headline.py --studies results/paper/studies
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "python" / "paper"))

import numpy as np

REFERENCE, NO_FILTER = "mhe_N7_o4c5", "mhe_N7_nofilter"


def rows_of(studies, variant):
    return list(csv.DictReader(open(studies / f"{variant}.csv")))


def by_participant(rows):
    table = {}
    for p in sorted({r["participant"] for r in rows}):
        mine = [r for r in rows if r["participant"] == p]
        table[p] = tuple(float(np.mean([float(r[c]) for r in mine]))
                         for c in ("causal_rmse_mean", "jitter_deg"))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", type=Path, required=True, help="folder of the study tables")
    args = ap.parse_args()
    from ik_filter_studies import ONE_EURO
    from rtcosmik.folds import FOLDS, fold_of
    from rtcosmik.paper import one_euro_selection as sel

    rows = {v: rows_of(args.studies, v) for v in ONE_EURO}
    candidates = {v: by_participant(r) for v, r in rows.items()}
    reference, no_filter = (by_participant(rows_of(args.studies, v)) for v in (REFERENCE, NO_FILTER))
    chosen = sel.choose_per_fold(candidates, reference)
    arm = sel.cross_validated(candidates, chosen)
    mean = lambda table, column: float(np.mean([v[column] for v in table.values()]))
    result = {"chosen_per_fold": chosen, "folds": [list(f) for f in FOLDS],
              "parameters": {v: ONE_EURO[v] for v in set(chosen) if v},
              "headline": sel.headline(arm, no_filter),
              "causal_deg": {"one_euro": arm and mean(arm, 0), "no_filter": mean(no_filter, 0),
                             "butterworth": mean(reference, 0)},
              "jitter_deg": {"one_euro": arm and mean(arm, 1), "no_filter": mean(no_filter, 1),
                             "butterworth": mean(reference, 1)}}
    (args.studies / "one_euro_choice.json").write_text(json.dumps(result, indent=2) + "\n")
    if arm is not None:
        kept = [dict(r, variant="one_euro") for v in ONE_EURO for r in rows[v]
                if chosen[fold_of(r["participant"])] == v]
        with open(args.studies / "one_euro.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
            writer.writeheader()
            writer.writerows(kept)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
