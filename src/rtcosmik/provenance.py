"""Provenance of a run: which code, settings, landmark map and machine produced it."""
import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Paths whose uncommitted changes would make a run untraceable.
PIPELINE_PATHS = ("src", "settings.py", "setup.py", "config", "scripts/python")


class DirtyTreeError(RuntimeError):
    """Pipeline code differs from the recorded commit."""


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def pipeline_changes(repo=REPO):
    """Uncommitted or untracked files under the pipeline paths."""
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--",
                  *PIPELINE_PATHS)
    return [line[3:] for line in status.splitlines() if line.strip()]


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        return repr(value)


def _hardware():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cpu": cpu, "gpu": gpu, "host": platform.node()}


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "torch", "pinocchio", "tensorrt"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def require_clean(repo=REPO):
    """Raise :class:`DirtyTreeError` if pipeline code has uncommitted changes."""
    changes = pipeline_changes(repo)
    if changes:
        raise DirtyTreeError(f"uncommitted pipeline changes in {repo}: {changes[:10]}")


def _indices_identity(indices):
    """Identity of the template vertices queried for the landmarks."""
    indices = [int(i) for i in indices]
    digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    return {"sha256": digest, "count": len(indices)}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def estimator_identity(settings, num_cameras=None):
    """NLF backend and detection mode of a run, with the engine files behind them."""
    backend = getattr(settings, "nlf_backend", "torchscript")
    mode = getattr(settings, "detection_mode", "per_view")
    if num_cameras == 1:
        mode = "per_view"
    identity = {"nlf_backend": backend, "detection_mode": mode}
    if mode == "tiled":
        from rtcosmik.model_weights import tiled_detector_engine
        imgsz = tuple(int(v) for v in settings.yolo_tiled_imgsz)
        identity["tiled_imgsz"] = list(imgsz)
        try:
            identity["detector_engine"] = Path(tiled_detector_engine(settings.yolo_path, imgsz)).name
        except (OSError, RuntimeError):
            identity["detector_engine"] = None
    if backend == "tensorrt":
        engine = Path(settings.nlf_engine_path)
        sidecar = Path(f"{engine}.json")
        identity["nlf_engine"] = {
            "file": engine.name,
            "sha256": sha256_of(engine) if engine.is_file() else None,
            "sidecar": json.loads(sidecar.read_text()) if sidecar.is_file() else None,
            "tensorrt_installed": _versions()["tensorrt"]}
    return identity


def recorded_estimator(views_path):
    """What made a views recording; recordings older than the stamp are TorchScript, per view."""
    import numpy as np
    data = np.load(views_path)
    if "estimator" in data:
        return json.loads(str(data["estimator"]))
    return {"nlf_backend": "torchscript", "detection_mode": "per_view"}


def collect_provenance(settings, arm, repo=REPO, landmark_map=None, estimator=None):
    """Provenance dict: date, commit, uncommitted pipeline files, arm, settings, map, machine.

    ``estimator`` is a replayed recording's :func:`recorded_estimator`; it replaces
    the replay-time backend and detection settings.
    """
    recorded = dict(vars(settings))
    if estimator is None:
        views = arm.get("num_cameras") or len(arm.get("cameras") or []) or None
        estimator = estimator_identity(settings, views)
    else:
        recorded.update(nlf_backend=estimator["nlf_backend"],
                        detection_mode=estimator["detection_mode"])
        if "tiled_imgsz" in estimator:
            recorded["yolo_tiled_imgsz"] = estimator["tiled_imgsz"]
    return {
        "started_utc": datetime.datetime.now(datetime.timezone.utc)
                       .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code": {"commit": _git(repo, "rev-parse", "HEAD"),
                 "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                 "uncommitted": pipeline_changes(repo)},
        "arm": _jsonable(dict(arm)),
        "settings": _jsonable(recorded),
        "estimator": estimator,
        "landmark_map": landmark_map or _indices_identity(getattr(settings, "nlf_indices", [])),
        "hardware": _hardware(),
        "versions": _versions(),
    }


def start_run(out_dir, settings, arm, repo=REPO, landmark_map=None, estimator=None):
    """Refuse uncommitted pipeline code, then write ``run_info.json``."""
    require_clean(repo)
    info = collect_provenance(settings, arm, repo, landmark_map, estimator)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    return info


def update_run_info(out_dir, **fields):
    """Add fields (frames, timings, ...) to an existing ``run_info.json``."""
    path = Path(out_dir) / "run_info.json"
    info = json.loads(path.read_text())
    info.update(_jsonable(fields))
    path.write_text(json.dumps(info, indent=2))
    return info
