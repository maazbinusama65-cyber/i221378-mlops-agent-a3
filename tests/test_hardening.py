"""Regression tests for the findings of the independent security/code review."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import httpx
import pytest

from agent.logging_utils import REDACTED, redact_text
from agent.schemas import DvcNoArgs, Status, ToolArgs
from agent.tools import airflow_tools, kubernetes_tools
from agent.tools.base import MAX_PENDING, Tool, ToolRegistry
from agent.tools.dvc_tools import _MD5_RE, _REV_RE
from tests.test_policy import evidence, verdict


# ---------------------------------------------------------------- secret redaction
@pytest.mark.parametrize("text,secret", [
    ('{"password": "hunter2-xyz"}', "hunter2-xyz"),
    ("password=hunter2-xyz", "hunter2-xyz"),
    ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA"),
    ("postgres://admin:s3cr3tpw@db.internal:5432/x", "s3cr3tpw"),
    ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnop"),
    ("key AKIAABCDEFGHIJKLMNOP leaked", "AKIAABCDEFGHIJKLMNOP"),
])
def test_common_secret_formats_are_redacted(text, secret):
    out = redact_text(text)
    assert secret not in out and REDACTED in out


def test_tool_output_is_redacted_before_it_reaches_the_model_or_ledger(registry: ToolRegistry):
    registry.register(Tool("leaky", "x", "d", DvcNoArgs, lambda a: {"log": "login ok Bearer abc.def.ghi password=pw12345"}))
    res = registry.execute("leaky", {})
    assert "abc.def.ghi" not in str(res.data) and "pw12345" not in str(res.data)
    assert "pw12345" not in registry.settings.log_file.read_text()


# ---------------------------------------------------------------- Airflow: path segments and conf
@pytest.fixture
def airflow(registry: ToolRegistry):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt-1"})
        return httpx.Response(200, json={"dag_runs": [], "dags": [], "task_instances": [], "dag_run_id": "r", "state": "queued"})

    client = airflow_tools.AirflowClient(registry.settings, transport=httpx.MockTransport(handler))
    registry.register_all(airflow_tools.build_tools(registry.settings, client))
    return registry, calls


@pytest.mark.parametrize("bad", ["x/../..", "~", "a?b=1", "../admin", "dag id", "a#b"])
def test_airflow_path_segments_cannot_redirect_requests(airflow, bad):
    registry, calls = airflow
    for tool, args in (("airflow_dag_runs", {"dag_id": bad}), ("airflow_task_instances", {"run_id": bad}),
                       ("airflow_task_logs", {"task_id": bad})):
        assert registry.execute(tool, args).status is Status.INVALID_REQUEST
    assert not any(path.startswith("/api/v2") for _, path in calls)


@pytest.mark.parametrize("conf", [{"evil": 1}, {"min_f1": 7}, {"n_estimators": 10**9}, {"algorithm": "rm -rf"},
                                  {"data_path": "../../etc/passwd"}, {"data_path": "/etc/passwd"}, {"data_path": "data/../secrets.csv"},
                                  {"data_path": "data/notes.txt"}])
def test_trigger_conf_is_restricted_to_safe_parameters(airflow, conf):
    registry, calls = airflow
    res = registry.execute("airflow_trigger_dag", {"conf": conf})
    assert res.status is Status.REJECTED and registry.pending == {}
    assert not any(m == "POST" for m, _ in calls)


def test_valid_trigger_conf_still_reaches_confirmation(airflow):
    registry, _ = airflow
    conf = {"min_f1": 0.6, "algorithm": "random_forest", "n_estimators": 100, "data_path": "data/failure_injection/churn_corrupt.csv"}
    res = registry.execute("airflow_trigger_dag", {"conf": conf})
    assert res.status is Status.CONFIRMATION_REQUIRED


# ---------------------------------------------------------------- registry robustness
class Num(ToolArgs):
    x: int = 0


def test_timed_out_write_reports_unknown_outcome(registry: ToolRegistry):
    registry.register(Tool("w", "x", "d", Num, lambda a: time.sleep(1.0) or {}, access="write", timeout_s=0.1))
    cid = registry.execute("w", {}).data["confirmation_id"]
    res = registry.confirm(cid)
    assert res.status is Status.TIMEOUT and "OUTCOME UNKNOWN" in res.error


def test_late_nested_calls_from_a_timed_out_tool_do_not_pollute_the_next_requests_ledger(registry: ToolRegistry):
    registry.register(Tool("inner", "x", "d", Num, lambda a: {"v": 1}))

    def slow_outer(_):
        time.sleep(0.4)
        registry.execute("inner", {})  # finishes after the outer call already timed out
        return {}

    registry.register(Tool("outer", "x", "d", Num, slow_outer, timeout_s=0.1))
    assert registry.execute("outer", {}).status is Status.TIMEOUT
    registry.reset_ledger()  # a new request begins
    time.sleep(0.7)
    assert registry.ledger == []  # the orphaned thread's late result was discarded
    assert registry.execute("inner", {}).evidence_id == "E1"  # ids restart cleanly per request


def test_pending_confirmations_are_bounded(registry: ToolRegistry):
    registry.register(Tool("w", "x", "d", Num, lambda a: {}, access="write"))
    for _ in range(MAX_PENDING):
        assert registry.execute("w", {}).status is Status.CONFIRMATION_REQUIRED
    assert registry.execute("w", {}).status is Status.REJECTED


def test_kubernetes_read_tools_still_work_with_epoch_binding(registry: ToolRegistry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    core.list_namespaced_pod.return_value = MagicMock(items=[])
    assert registry.execute("k8s_list_pods", {}).ok


# ---------------------------------------------------------------- input validation + policy
def test_revision_and_md5_validation_reject_trailing_newlines_and_options():
    assert _MD5_RE.fullmatch("a" * 32) and not _MD5_RE.fullmatch("a" * 32 + "\n")
    assert _REV_RE.fullmatch("data-v1") and not _REV_RE.fullmatch("data-v1\n") and not _REV_RE.fullmatch("--output=x")


def test_missing_dataset_version_in_configmap_blocks_a_no_retraining_verdict():
    ev = evidence()
    del ev.deployment["config"]["DATASET_VERSION"]
    out = verdict(ev)
    assert out["verdict"] == "insufficient_evidence"
    assert any("DATASET_VERSION missing" in u for u in out["uncertainties"])
