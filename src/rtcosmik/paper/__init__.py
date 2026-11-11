"""Code that exists only to produce the RT-COSMIK paper's comparative results.

Nothing here is part of the live toolbox. It is kept apart so the pipeline
proper stays free of baseline-specific special cases.
"""


def require_paper_evaluation(settings):
    """Refuse to produce paper results unless settings.py selects the paper evaluation."""
    if not getattr(settings, "paper_evaluation", False):
        raise SystemExit("settings.paper_evaluation is False: paper results need the paper "
                         "configuration (parity markers, fitted vertices, 5 Hz, N = 7).")
