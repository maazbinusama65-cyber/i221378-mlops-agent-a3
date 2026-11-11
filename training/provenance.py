"""Dataset / code provenance recorded on every training run.

Pure-Python on purpose: it must work inside the Airflow container, which has no
``git`` or ``dvc`` binaries but does see the mounted repository.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - DVC itself identifies content with md5
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dvc_file(data_path: Path) -> dict:
    dvc_file = data_path.with_name(data_path.name + ".dvc")
    if not dvc_file.exists():
        return {}
    return yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}


def git_commit(repo_root: Path) -> str:
    """Resolve HEAD without invoking git (reads .git directly)."""
    override = os.environ.get("GIT_COMMIT")
    if override:
        return override
    git_dir = repo_root / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return "unknown"


def dataset_provenance(data_path: Path, repo_root: Path) -> dict[str, str]:
    """Return the tags that tie a run to an exact dataset version and code commit."""
    dvc = read_dvc_file(data_path)
    recorded = (dvc.get("outs") or [{}])[0].get("md5", "unknown")
    actual = file_md5(data_path) if data_path.exists() else "missing"
    meta = dvc.get("meta") or {}
    return {
        "dvc.dataset_version": str(meta.get("version", "untracked")),
        "dvc.md5": actual,
        "dvc.recorded_md5": recorded,
        "dvc.dirty": str(actual != recorded).lower(),
        "dvc.dataset_path": data_path.as_posix(),
        "dvc.remote": "localstore",
        "git.commit": git_commit(repo_root),
    }
