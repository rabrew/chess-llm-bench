"""Provenance for summary.json: code, data, engine and settings (specs/summary-provenance.md)."""

import hashlib
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .utils import load_config

logger = logging.getLogger("chess_llm_bench")

HASH_CHUNK_BYTES = 1 << 20
GIT_TIMEOUT_S = 10
TRACKED_PACKAGES = ("pandas", "numpy", "chess")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        logger.warning(f"Provenance: {path} not found; no hash recorded")
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Provenance: git {' '.join(args)} failed: {exc}")
        return None
    if result.returncode != 0:
        logger.warning(f"Provenance: git {' '.join(args)} failed: {result.stderr.strip()}")
        return None
    return result.stdout


def _git_state(repo_dir: Path) -> tuple[str | None, bool | None]:
    sha = _git(repo_dir, "rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git(repo_dir, "status", "--porcelain", "--untracked-files=no")
    return sha.strip(), (None if status is None else bool(status.strip()))


def _engine(config: dict) -> dict[str, Any] | None:
    lc0 = config.get("lc0")
    if not lc0:
        return None
    weights = lc0.get("weights")
    return {
        "name": "lc0",
        "binary": lc0.get("binary"),
        "weights": weights,
        "weights_sha256": _sha256(Path(weights)) if weights else None,
        "nodes": lc0.get("nodes"),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version("python-chess" if name == "chess" else name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_provenance(config_path: str | Path, results_file: str | Path, repo_dir: str | Path = ".") -> dict[str, Any]:
    """Everything needed to trace a summary back to the run that produced it."""
    config = load_config(str(config_path))
    results_path = Path(results_file)
    git_sha, git_dirty = _git_state(Path(repo_dir))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "results_file": str(results_file),
        "results_sha256": _sha256(results_path),
        "results_records": _line_count(results_path),
        "engine": _engine(config),
        "config_path": str(config_path),
        "config": config,
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
