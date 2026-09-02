"""Resolving model artefacts on disk for the running configuration.

Settings name a model generically; which file is actually loaded can depend on
the runtime setup. The detector is the case in point: TensorRT engines are
exported non-dynamic, so an engine only accepts the exact batch size it was
built for. ``scripts/bash/fetch_models.sh`` builds one engine per supported
camera count, named ``<stem>_b<N><suffix>``, and this module picks the one
matching the cameras in use.

Kept next to config_loader rather than under nlf/ because it resolves detector
weights, not NLF weights, and is used by the pipeline as well.
"""

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def engine_path_for(base_path, num_cameras):
    """Return the per-camera-count engine path derived from a base path."""
    base = Path(base_path)
    return base.with_name(f"{base.stem}_b{num_cameras}{base.suffix}")


def engine_batch_size(engine_path):
    """Read the batch size an engine was built for, or None if unrecorded."""
    meta_path = Path(f"{engine_path}.meta")
    if not meta_path.is_file():
        return None
    try:
        return int(meta_path.read_text().strip().split(",")[0])
    except (OSError, ValueError, IndexError):
        return None


def available_camera_counts(base_path):
    """Camera counts that have an engine built for them, sorted."""
    base = Path(base_path)
    if not base.parent.is_dir():
        return []
    counts = []
    for candidate in base.parent.glob(f"{base.stem}_b*{base.suffix}"):
        batch = engine_batch_size(candidate)
        if batch is not None:
            counts.append(batch)
    return sorted(counts)


def resolve_detector_engine(base_path, num_cameras):
    """
    Pick the detector engine built for ``num_cameras``.

    Falls back to ``base_path`` when no per-count engine exists, so an install
    predating the per-count layout keeps working as long as its single engine
    was built for the right number of cameras.

    Raises:
        FileNotFoundError: no engine matches and the fallback is unusable.
    """
    candidate = engine_path_for(base_path, num_cameras)
    if candidate.is_file():
        recorded = engine_batch_size(candidate)
        if recorded not in (None, num_cameras):
            raise RuntimeError(
                f"{candidate} is named for {num_cameras} camera(s) but records batch={recorded}. "
                f"Rebuild it with: BATCHES={num_cameras} bash scripts/bash/fetch_models.sh"
            )
        LOGGER.info("Using detector engine for %d camera(s): %s", num_cameras, candidate)
        return str(candidate)

    base = Path(base_path)
    if base.is_file():
        recorded = engine_batch_size(base)
        if recorded in (None, num_cameras):
            LOGGER.info("Using detector engine %s for %d camera(s)", base, num_cameras)
            return str(base)

    available = available_camera_counts(base_path)
    hint = (f"engines are available for {available} camera(s)"
            if available else "no per-camera-count engines were found")
    raise FileNotFoundError(
        f"No detector engine for {num_cameras} camera(s) ({hint}).\n"
        f"Build one with: BATCHES={num_cameras} bash scripts/bash/fetch_models.sh"
    )
