"""Choice of the 1-euro filter's parameters per fold, and the headline filter rule.

Tables map participant -> (causal error, jitter), each a mean over the participant's trials.
"""
import numpy as np

from rtcosmik.folds import FOLDS, fitting_participants, fold_of


def _mean(table, participants, column):
    return float(np.mean([table[p][column] for p in participants]))


def choose_per_fold(candidates, reference):
    """For each fold, the candidate with the lowest causal error on the fold's fitting
    participants among those whose jitter there does not exceed the reference's; None if none."""
    chosen = []
    for fold in range(len(FOLDS)):
        fit = fitting_participants(fold)
        limit = _mean(reference, fit, 1)
        allowed = [name for name, t in candidates.items() if _mean(t, fit, 1) <= limit]
        chosen.append(min(allowed, key=lambda name: _mean(candidates[name], fit, 0)) if allowed else None)
    return chosen


def cross_validated(candidates, chosen):
    """Each participant's row under its own fold's choice; None when a fold has no choice."""
    if any(name is None for name in chosen):
        return None
    participants = next(iter(candidates.values()))
    return {p: candidates[chosen[fold_of(p)]][p] for p in participants}


def headline(one_euro, no_filter):
    """The headline arm: MHE without filter unless the 1-euro arm has lower causal error."""
    if one_euro is None:
        return "no_filter"
    participants = sorted(no_filter)
    return "one_euro" if _mean(one_euro, participants, 0) < _mean(no_filter, participants, 0) else "no_filter"
