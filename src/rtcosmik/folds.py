"""The six fixed folds of COMFI participants (IDs only).

Fold k holds every sixth participant of the sorted ID list, starting at k.
"""

PARTICIPANTS = ("1012", "1118", "1508", "1602", "1847", "2112", "2198", "2307", "3361",
                "4162", "4216", "4279", "4509", "4612", "4665", "4687", "4801", "4827")

FOLDS = (
    ("1012", "2198", "4509"),
    ("1118", "2307", "4612"),
    ("1508", "3361", "4665"),
    ("1602", "4162", "4687"),
    ("1847", "4216", "4801"),
    ("2112", "4279", "4827"),
)

assert sorted(p for fold in FOLDS for p in fold) == list(PARTICIPANTS)


def fold_of(participant):
    """Index of the fold holding ``participant``."""
    for index, fold in enumerate(FOLDS):
        if str(participant) in fold:
            return index
    raise KeyError(f"participant {participant} is in no fold")


def fitting_participants(fold):
    """The fifteen participants a fold's choices may be fitted on."""
    return tuple(p for p in PARTICIPANTS if p not in FOLDS[fold])
