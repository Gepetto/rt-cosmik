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
    for name in ("numpy", "torch", "pinocchio"):
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


def landmark_map(indices):
    """Identity of the template vertices queried for the landmarks."""
    indices = [int(i) for i in indices]
    digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    return {"sha256": digest, "count": len(indices)}


def collect_provenance(settings, arm, repo=REPO):
    """Provenance dict: date, commit, uncommitted pipeline files, arm, settings, machine."""
    return {
        "started_utc": datetime.datetime.now(datetime.timezone.utc)
                       .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code": {"commit": _git(repo, "rev-parse", "HEAD"),
                 "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                 "uncommitted": pipeline_changes(repo)},
        "arm": _jsonable(dict(arm)),
        "settings": _jsonable(dict(vars(settings))),
        "landmark_map": landmark_map(getattr(settings, "nlf_indices", [])),
        "hardware": _hardware(),
        "versions": _versions(),
    }


def start_run(out_dir, settings, arm, repo=REPO):
    """Refuse uncommitted pipeline code, then write ``run_info.json``."""
    require_clean(repo)
    info = collect_provenance(settings, arm, repo)
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
