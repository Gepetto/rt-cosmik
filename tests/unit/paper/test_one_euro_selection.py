"""Per-fold choice of the 1-euro parameters and the headline filter rule."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.folds import FOLDS, PARTICIPANTS
from rtcosmik.paper import one_euro_selection as sel


def table(causal, jitter):
    """{participant: (causal, jitter)} with the same values for everyone, or per participant."""
    value = lambda v, p: v[p] if isinstance(v, dict) else v
    return {p: (value(causal, p), value(jitter, p)) for p in PARTICIPANTS}


def test_lowest_causal_error_among_candidates_no_more_jittery_than_the_reference():
    candidates = {"smooth": table(9.0, 0.15), "sharp": table(8.0, 0.30), "ok": table(8.5, 0.16)}
    chosen = sel.choose_per_fold(candidates, reference=table(9.2, 0.16))
    assert chosen == ["ok"] * len(FOLDS)


def test_choice_ignores_the_held_out_participants():
    held = set(FOLDS[0])
    good_only_there = {p: 1.0 if p in held else 9.0 for p in PARTICIPANTS}
    candidates = {"a": table(8.0, 0.1), "b": table(good_only_there, 0.1)}
    assert sel.choose_per_fold(candidates, reference=table(9.0, 0.2))[0] == "a"


def test_fold_with_no_admissible_candidate_has_no_choice():
    assert sel.choose_per_fold({"sharp": table(8.0, 0.3)}, reference=table(9.0, 0.2)) == [None] * len(FOLDS)


def test_cross_validated_arm_takes_each_participant_from_its_folds_choice():
    candidates = {"a": table(8.0, 0.1), "b": table(7.0, 0.1)}
    chosen = ["a"] + ["b"] * (len(FOLDS) - 1)
    arm = sel.cross_validated(candidates, chosen)
    assert arm[FOLDS[0][0]] == (8.0, 0.1) and arm[FOLDS[1][0]] == (7.0, 0.1)
    assert sel.cross_validated(candidates, [None] + ["b"] * 5) is None


def test_headline_is_no_filter_unless_one_euro_has_lower_causal_error():
    no_filter = table(8.45, 0.28)
    assert sel.headline(table(8.40, 0.15), no_filter) == "one_euro"
    assert sel.headline(table(8.50, 0.15), no_filter) == "no_filter"
    assert sel.headline(None, no_filter) == "no_filter"
