"""DVC tools against a real throw-away git+DVC repository (no network, local cache only)."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.config import Settings
from agent.schemas import Status
from agent.tools import dvc_tools
from agent.tools.base import ToolRegistry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@example.com", "DVC_NO_ANALYTICS": "1", "PYTHONUTF8": "1"}


def sh(cwd: Path, *argv: str) -> None:
    subprocess.run(argv, cwd=cwd, env=ENV, check=True, capture_output=True)


def dvc(cwd: Path, *args: str) -> None:
    sh(cwd, sys.executable, "-m", "dvc", *args)


def write_version(repo: Path, version: str, content: str, tag: str) -> None:
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "churn.csv").write_text(content)
    dvc(repo, "add", "data/churn.csv")
    with (repo / "data" / "churn.csv.dvc").open("a") as fh:
        fh.write(f"meta:\n  version: {version}\n")
    sh(repo, "git", "add", "-A")
    sh(repo, "git", "commit", "-m", f"data {version}")
    sh(repo, "git", "tag", tag)


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dvcrepo")
    sh(root, "git", "init", "-b", "main")
    dvc(root, "init")
    sh(root, "git", "commit", "-m", "init dvc")
    write_version(root, "v1", "a,b\n1,2\n3,4\n", "data-v1")
    write_version(root, "v2", "a,b\n1,2\n3,4\n5,6\n", "data-v2")
    return root


@pytest.fixture
def dvc_registry(repo: Path, tmp_path) -> ToolRegistry:
    settings = dataclasses.replace(Settings(), repo_root=repo, log_file=tmp_path / "agent.jsonl", tool_timeout_s=60.0)
    registry = ToolRegistry(settings)
    registry.register_all(dvc_tools.build_tools(settings))
    return registry


def test_current_version_and_git_commit_come_from_recorded_provenance(dvc_registry):
    data = dvc_registry.execute("dvc_dataset_version", {}).data
    assert data["dataset_version"] == "v2"
    assert data["dataset_commit"]["tags"] == ["data-v2"] and len(data["dataset_commit"]["commit"]) == 40


def test_list_datasets_finds_tracked_files(dvc_registry):
    data = dvc_registry.execute("dvc_list_datasets", {}).data
    assert [d["path"] for d in data["datasets"]] == ["data/churn.csv"] and data["datasets"][0]["version"] == "v2"


def test_resolve_reference_maps_md5_back_to_version_and_commit(dvc_registry):
    v1 = dvc_registry.execute("dvc_resolve_reference", {"version": "v1"}).data
    assert v1["found"] and v1["dataset_commit"]["tags"] == ["data-v1"]
    again = dvc_registry.execute("dvc_resolve_reference", {"md5": v1["md5"]}).data
    assert again["version"] == "v1" and again["reproduce_with_tag"] == "data-v1"
    assert dvc_registry.execute("dvc_resolve_reference", {"md5": "0" * 32}).data["found"] is False


def test_compare_versions_detects_content_change(dvc_registry):
    data = dvc_registry.execute("dvc_compare_versions", {"rev_a": "data-v1", "rev_b": "data-v2"}).data
    assert data["content_changed"] is True and data["size_delta_bytes"] > 0
    same = dvc_registry.execute("dvc_compare_versions", {"rev_a": "data-v2", "rev_b": "data-v2"}).data
    assert same["content_changed"] is False


def test_change_detection_notices_modified_data(dvc_registry, repo: Path):
    assert dvc_registry.execute("dvc_check_changes", {}).data["data_changed"] is False
    target = repo / "data" / "churn.csv"
    original = target.read_text()
    try:
        target.write_text(original + "9,9\n")
        assert dvc_registry.execute("dvc_check_changes", {}).data["data_changed"] is True
    finally:
        target.write_text(original)


def test_reproduce_verifies_old_version_in_sandbox_without_touching_working_tree(dvc_registry, repo: Path):
    before = (repo / "data" / "churn.csv").read_text()
    data = dvc_registry.execute("dvc_explain_reproduce", {"rev": "data-v1", "verify": True}).data
    assert data["verification"]["matches_recorded_md5"] is True
    assert (repo / "data" / "churn.csv").read_text() == before
    assert any("dvc checkout" in step for step in data["steps"])


def test_unapproved_paths_and_option_injection_are_rejected(dvc_registry):
    assert dvc_registry.execute("dvc_dataset_version", {"path": "../../etc/passwd"}).status is Status.REJECTED
    assert dvc_registry.execute("dvc_compare_versions", {"rev_a": "--output=/tmp/x", "rev_b": "data-v2"}).status is Status.REJECTED


def test_only_allowlisted_commands_can_run(repo: Path):
    from agent.tools.base import ToolRejected

    with pytest.raises(ToolRejected):
        dvc_tools.run_dvc(repo, "push")
    with pytest.raises(ToolRejected):
        dvc_tools.run_git(repo, "push", "origin", "main")
