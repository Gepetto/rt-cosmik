# Causal, out-of-fold evaluation on the paper branch

The TII resubmission reports **causal error** (no time alignment) as the headline metric, with aligned error and lag beside it, because the paper's earlier aligned numbers hid the filter and pipeline lag that a real-time user experiences. Every choice fitted against the reference (landmark maps for NLF and FastSAM-3D, 1€ filter parameters) is made per **fold**: six disjoint groups of three COMFI participants, each evaluated with choices fitted on the other fifteen, so all 18 participants are out-of-fold and paired tests keep n = 18. The work branches from `paper/mmpose-baseline` (`36c7725`, whose pipeline code equals the paper run `25b7db5`), not from `main`, because `main` was 57 commits behind the code that produced the submitted numbers.

## Considered Options

- Single split (6 development, 12 test): cheaper by a third of one NLF pass, but n = 12.
- Keep aligned error as headline: rejected by R2.7 and R3.3; it makes the low-pass filter look free.
- Branch from `main`: lacked the causal filter fix, the 3361 camera relabel and the COMFI loaders.
