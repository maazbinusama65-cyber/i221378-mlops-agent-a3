"""DVC data-management tools.

Safety model
  * Only an allowlist of read-oriented DVC / git sub-commands can run (``ALLOWED_DVC`` / ``ALLOWED_GIT``).
  * Commands run via ``subprocess`` with an argument list - never through a shell.
  * Dataset paths must be on the configured allowlist; revisions are validated so they cannot smuggle options.
  * The "checkout" demonstration uses ``dvc get`` into a sandbox directory, so the working tree is never touched.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent.config import Settings
from agent.schemas import DvcCompareArgs, DvcDatasetArgs, DvcNoArgs, DvcReproduceArgs, DvcResolveArgs
from agent.tools.base import Tool, ToolError, ToolRejected, ToolUnavailable
from training.provenance import file_md5

ALLOWED_DVC = {"status", "diff", "get", "list", "data", "version"}
ALLOWED_GIT = {"log", "show", "rev-parse", "tag", "describe", "status", "ls-files"}
_REV_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-~^]*")
_MD5_RE = re.compile(r"[0-9a-f]{32}")
SANDBOX = ".agent-sandbox"


def _run(argv: list[str], cwd: Path, timeout: float = 25.0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DVC_NO_ANALYTICS": "1", "PYTHONUTF8": "1", "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, check=False, shell=False)  # noqa: S603
    except FileNotFoundError as exc:
        raise ToolUnavailable(f"{argv[0]} not installed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"{' '.join(argv[:3])} timed out after {timeout}s") from exc


def run_git(cwd: Path, *args: str) -> str:
    if not args or args[0] not in ALLOWED_GIT:
        raise ToolRejected(f"git sub-command not approved: {args[:1]}")
    proc = _run(["git", *args], cwd)
    if proc.returncode != 0:
        raise ToolError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def run_dvc(cwd: Path, *args: str, timeout: float = 25.0) -> subprocess.CompletedProcess[str]:
    if not args or args[0] not in ALLOWED_DVC:
        raise ToolRejected(f"dvc sub-command not approved: {args[:1]}")
    return _run([sys.executable, "-m", "dvc", *args], cwd, timeout)


def _check_rev(rev: str) -> str:
    if not _REV_RE.fullmatch(rev):
        raise ToolRejected(f"invalid git revision {rev!r}")
    return rev


def _parse_dvc(text: str) -> dict[str, Any]:
    doc = yaml.safe_load(text) or {}
    out = (doc.get("outs") or [{}])[0]
    meta = doc.get("meta") or {}
    return {"md5": out.get("md5"), "size": out.get("size"), "version": meta.get("version"), "description": meta.get("description")}


class DvcTools:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.root = settings.repo_root

    # -- helpers ---------------------------------------------------------------------------
    def _dataset(self, path: str) -> str:
        norm = path.replace("\\", "/")
        if norm not in self.s.dvc_datasets:
            raise ToolRejected(f"dataset path {path!r} is not on the approved list {list(self.s.dvc_datasets)}")
        return norm

    def _dvc_file_at(self, rev: str, path: str) -> dict[str, Any]:
        return _parse_dvc(run_git(self.root, "show", f"{_check_rev(rev)}:{path}.dvc"))

    def _commit_info(self, commit: str) -> dict[str, Any]:
        line = run_git(self.root, "log", "-1", "--format=%H|%h|%cI|%s", commit)
        full, short, date, subject = line.split("|", 3)
        tags = run_git(self.root, "tag", "--points-at", full).split()
        return {"commit": full, "short": short, "date": date, "subject": subject, "tags": tags}

    def _remote(self) -> dict[str, Any]:
        cfg = configparser.ConfigParser()
        cfg.read(self.root / ".dvc" / "config")
        for section in cfg.sections():
            if section.startswith('\'remote "') or section.startswith('"remote'):
                continue
        default = cfg.get("core", "remote", fallback=None)
        url = cfg.get(f"'remote \"{default}\"'", "url", fallback=None) if default else None
        return {"name": default, "url": url}

    def _in_remote(self, md5: str) -> bool | None:
        url = self._remote()["url"]
        if not url or "://" in url:
            return None  # only local remotes can be checked without network access
        base = (self.root / ".dvc" / url).resolve()
        return (base / "files" / "md5" / md5[:2] / md5[2:]).exists()

    # -- tools -----------------------------------------------------------------------------
    def list_datasets(self, _: DvcNoArgs) -> dict[str, Any]:
        found = []
        listing = run_git(self.root, "ls-files", "--cached", "--others", "--exclude-standard", "*.dvc")
        for rel_name in sorted(filter(None, listing.splitlines())):
            dvc_file = self.root / rel_name
            if not dvc_file.is_file():
                continue
            info = _parse_dvc(dvc_file.read_text(encoding="utf-8"))
            found.append({"dvc_file": rel_name, "path": rel_name[:-4], **info})
        return {"datasets": found, "count": len(found), "remote": self._remote()}

    def dataset_version(self, args: DvcDatasetArgs) -> dict[str, Any]:
        path = self._dataset(args.path)
        dvc_rel = f"{path}.dvc"
        info = _parse_dvc((self.root / dvc_rel).read_text(encoding="utf-8"))
        commit = self._commit_info(run_git(self.root, "log", "-1", "--format=%H", "--", dvc_rel))
        head = run_git(self.root, "rev-parse", "HEAD")
        return {
            "path": path,
            "dataset_version": info["version"],
            "md5": info["md5"],
            "size_bytes": info["size"],
            "dataset_commit": commit,
            "git_head": head,
            "remote": self._remote(),
            "in_remote": self._in_remote(info["md5"]) if info["md5"] else None,
        }

    def resolve_reference(self, args: DvcResolveArgs) -> dict[str, Any]:
        if not (args.md5 or args.version):
            raise ToolError("provide md5 or version")
        if args.md5 and not _MD5_RE.fullmatch(args.md5):
            raise ToolRejected(f"invalid md5 {args.md5!r}")
        path = self._dataset(args.path)
        history = run_git(self.root, "log", "--format=%H", "--", f"{path}.dvc").split()
        for commit in history:  # newest first
            try:
                info = _parse_dvc(run_git(self.root, "show", f"{commit}:{path}.dvc"))
            except ToolError:
                continue
            if (args.md5 and info["md5"] == args.md5) or (args.version and info["version"] == args.version):
                result = {"found": True, "path": path, **info, "dataset_commit": self._commit_info(commit)}
                result["in_remote"] = self._in_remote(info["md5"]) if info["md5"] else None
                result["reproduce_with_tag"] = (result["dataset_commit"]["tags"] or [commit[:12]])[0]
                return result
        return {"found": False, "path": path, "searched_commits": len(history), "md5": args.md5, "version": args.version}

    def compare_versions(self, args: DvcCompareArgs) -> dict[str, Any]:
        path = self._dataset(args.path)
        a, b = self._dvc_file_at(args.rev_a, path), self._dvc_file_at(args.rev_b, path)
        diff_out = run_dvc(self.root, "diff", "--json", _check_rev(args.rev_a), _check_rev(args.rev_b), "--", path)
        diff = None
        if diff_out.returncode == 0:
            try:
                diff = json.loads(diff_out.stdout or "{}")
            except ValueError:
                diff = None
        changed = a["md5"] != b["md5"]
        return {
            "path": path,
            "a": {"rev": args.rev_a, **a},
            "b": {"rev": args.rev_b, **b},
            "content_changed": changed,
            "size_delta_bytes": (b["size"] or 0) - (a["size"] or 0),
            "dvc_diff": diff,
            "summary": f"{args.rev_a} ({a['version']}) -> {args.rev_b} ({b['version']}): "
            + ("data content differs" if changed else "identical content"),
        }

    def check_changes(self, args: DvcDatasetArgs) -> dict[str, Any]:
        path = self._dataset(args.path)
        info = _parse_dvc((self.root / f"{path}.dvc").read_text(encoding="utf-8"))
        file = self.root / path
        actual = file_md5(file) if file.exists() else None
        dvc_status = run_dvc(self.root, "status", f"{path}.dvc", "--json")
        dvc_changed = dvc_status.stdout.strip() not in ("", "{}")
        git_dirty = bool(run_git(self.root, "status", "--porcelain", "--", f"{path}.dvc"))
        return {
            "path": path,
            "recorded_md5": info["md5"],
            "working_md5": actual,
            "data_changed": actual != info["md5"],
            "dvc_status_reports_change": dvc_changed,
            "dvc_file_uncommitted": git_dirty,
            "file_missing": actual is None,
        }

    def explain_reproduce(self, args: DvcReproduceArgs) -> dict[str, Any]:
        path = self._dataset(args.path)
        info = self._dvc_file_at(args.rev, path)
        steps = [
            f"git checkout {args.rev} -- {path}.dvc      # restore the recorded dataset pointer",
            f"dvc checkout {path}.dvc                    # materialise data from the DVC cache (or `dvc pull` first)",
            f"md5sum {path}                              # must equal {info['md5']}",
            f"git checkout HEAD -- {path}.dvc && dvc checkout {path}.dvc   # return to the current dataset",
        ]
        result: dict[str, Any] = {
            "rev": args.rev, "version": info["version"], "expected_md5": info["md5"],
            "non_destructive_alternative": f"dvc get . {path} --rev {args.rev} -o <dir>",
            "steps": steps, "in_remote": self._in_remote(info["md5"]) if info["md5"] else None,
        }
        if args.verify:
            out_dir = self.root / SANDBOX / args.rev.replace("/", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / Path(path).name
            proc = run_dvc(self.root, "get", ".", path, "--rev", _check_rev(args.rev), "-o", str(target), "--force", timeout=60)
            if proc.returncode != 0:
                raise ToolError(f"dvc get failed: {(proc.stderr or proc.stdout).strip()[:300]}")
            actual = file_md5(target)
            result["verification"] = {"exported_to": target.relative_to(self.root).as_posix(), "actual_md5": actual,
                                      "matches_recorded_md5": actual == info["md5"]}
        return result


def build_tools(settings: Settings) -> list[Tool]:
    d = DvcTools(settings)
    return [
        Tool("dvc_list_datasets", "dvc", "List datasets tracked by DVC with md5, size and version label.", DvcNoArgs, d.list_datasets),
        Tool("dvc_dataset_version", "dvc", "Current dataset version, md5 and the Git commit that recorded it.", DvcDatasetArgs, d.dataset_version),
        Tool("dvc_resolve_reference", "dvc",
             "Find the Git commit/tag/version for a dataset md5 (from an MLflow run's dvc.md5 tag) or a version label.",
             DvcResolveArgs, d.resolve_reference),
        Tool("dvc_compare_versions", "dvc", "Compare two dataset versions (git revs/tags): md5, size, DVC diff.", DvcCompareArgs, d.compare_versions),
        Tool("dvc_check_changes", "dvc", "Detect whether the tracked data in the working tree changed vs its recorded md5.", DvcDatasetArgs, d.check_changes,
             timeout_s=60.0),
        Tool("dvc_explain_reproduce", "dvc",
             "Explain how to reproduce a dataset version; verify=true safely exports it to a sandbox and checks the md5.",
             DvcReproduceArgs, d.explain_reproduce, timeout_s=90.0),
    ]
