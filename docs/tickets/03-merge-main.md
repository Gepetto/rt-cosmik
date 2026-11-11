# 03: Merge main into the resubmission branch

**What to build:** The resubmission branch sits on top of the current toolbox on main (documentation, Docker image, pinned versions, CI) while still producing the paper's results. Main's new defaults (10 Hz cutoff, N = 10, hand-picked NLF vertices, no parity marker set) no longer decide paper numbers: the paper configuration is stated explicitly in the arm descriptions. A trial merge showed four conflicts (settings, pipeline, COMFI scene, gitignore) and one paper script relying on a filtering helper main removed.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Main merged; the four conflicts resolved
- [ ] Paper configuration (5 Hz, N = 7, fitted NLF map, parity marker set, locked joints) expressed in arm descriptions, not defaults
- [ ] The removed filtering helper restored or replaced for the paper script that uses it
- [ ] Ticket 02's regression test reproduces the submitted numbers for one participant after the merge
- [ ] No commit or push without the author's authorization
