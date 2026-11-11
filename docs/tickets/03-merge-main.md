# 03: Merge main into the resubmission branch

**What to build:** The resubmission branch sits on top of the current toolbox on main (documentation, Docker image, pinned versions, CI) while still producing the paper's results. Main's new defaults (10 Hz cutoff, N = 10, hand-picked NLF vertices, no parity marker set) no longer decide paper numbers: the paper configuration is stated explicitly in the arm descriptions. A trial merge showed four conflicts (settings, pipeline, COMFI scene, gitignore) and one paper script relying on a filtering helper main removed.

**Blocked by:** 02

**Status:** merged in the working tree, awaiting the author's commit and regression run

- [x] Main merged; the four conflicts resolved
- [x] Paper configuration (5 Hz, N = 7, fitted NLF map, parity marker set, locked joints) stated in settings.py behind one switch, `paper_evaluation`, on top of main's toolbox defaults
- [x] The removed filtering helper: false alarm, it is defined locally in the vendored viz tool and only used in a comment
- [ ] Ticket 02's regression test reproduces the submitted numbers for one participant after the merge
- [ ] No commit or push without the author's authorization

## Notes

- settings.py holds the paper configuration (decision A): `paper_evaluation = True` applies the `paper` block (parity markers, fitted NLF vertices, 5 Hz, order 4, N = 7) on top of main's toolbox defaults; `False` gives main's toolbox behaviour. The fallback settings in config_loader mirror it, as main's test requires.
- The marker-set mechanics live in `rtcosmik.marker_sets`; paper runners refuse to run when `paper_evaluation` is False.
- One settings object: solver generation and scoring (locked DoF excluded, 29 scored) see the paper configuration. The arm test checks the 29 scored DoF; the regression test also compares the score with the submitted run's.
