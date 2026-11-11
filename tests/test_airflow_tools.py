"""Airflow tools against a mocked REST API (httpx.MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from agent.schemas import Status
from agent.tools import airflow_tools
from agent.tools.base import ToolRegistry

RUN = {"dag_run_id": "manual__1", "state": "failed", "run_type": "manual", "logical_date": None,
       "start_date": "2026-01-01T00:00:00Z", "end_date": "2026-01-01T00:01:00Z", "triggered_by": "rest_api", "conf": {}}
TASKS = [
    {"task_id": "validate_data", "state": "success", "try_number": 1, "max_tries": 2, "duration": 1.2},
    {"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 2, "duration": 0.4},
    {"task_id": "register_candidate", "state": "upstream_failed", "try_number": 0, "max_tries": 2, "duration": None},
]


class FakeAirflow:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.token_requests = 0
        self.expire_next = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/auth/token":
            self.token_requests += 1
            return httpx.Response(201, json={"access_token": f"jwt-{self.token_requests}"})
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(401, json={"detail": "expired"})
        if not request.headers.get("authorization", "").startswith("Bearer jwt-"):
            return httpx.Response(401)
        if path == "/api/v2/dags":
            return httpx.Response(200, json={"dags": [{"dag_id": "churn_training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]},
                                                      {"dag_id": "other_dag", "is_paused": False, "tags": []}]})
        if path.endswith("/dagRuns") and request.method == "GET":
            return httpx.Response(200, json={"dag_runs": [RUN]})
        if path.endswith("/dagRuns") and request.method == "POST":
            return httpx.Response(200, json={**RUN, "state": "queued"})
        if path.endswith("/taskInstances"):
            return httpx.Response(200, json={"task_instances": TASKS})
        if "/logs/" in path:
            return httpx.Response(200, json={"content": [
                {"timestamp": "2026-01-01T00:00:30Z", "level": "error", "event": "QUALITY GATE FAILED for run abc: f1 0.31 < min_f1 0.55",
                 "error_detail": [{"exc_type": "AirflowFailException", "exc_value": "quality gate failed: f1 0.31 < min_f1 0.55"}]}]})
        return httpx.Response(404)


@pytest.fixture
def airflow(registry: ToolRegistry):
    fake = FakeAirflow()
    client = airflow_tools.AirflowClient(registry.settings, transport=httpx.MockTransport(fake))
    registry.register_all(airflow_tools.build_tools(registry.settings, client))
    return registry, fake


def test_lists_dags_with_approval_flags_using_jwt_auth(airflow):
    registry, fake = airflow
    res = registry.execute("airflow_list_dags", {})
    approved = {d["dag_id"]: d["approved_to_trigger"] for d in res.data["dags"]}
    assert approved == {"churn_training_pipeline": True, "other_dag": False}
    assert fake.token_requests == 1


def test_expired_token_is_refreshed_transparently(airflow):
    registry, fake = airflow
    registry.execute("airflow_dag_runs", {})
    fake.expire_next = True
    assert registry.execute("airflow_dag_runs", {}).status is Status.OK
    assert fake.token_requests == 2


def test_failed_tasks_identify_root_cause_and_include_log_diagnostics(airflow):
    registry, _ = airflow
    data = registry.execute("airflow_failed_tasks", {}).data
    assert data["root_cause_task"] == "quality_gate"
    assert [t["task_id"] for t in data["failed_tasks"]] == ["quality_gate", "register_candidate"]
    assert any("EXCEPTION AirflowFailException" in line for line in data["failed_tasks"][0]["log_tail"])
    assert "register_candidate=upstream_failed" in data["all_tasks"]


def test_triggering_requires_confirmation_and_only_approved_dags(airflow):
    registry, fake = airflow
    unapproved = registry.execute("airflow_trigger_dag", {"dag_id": "other_dag"})
    assert unapproved.status is Status.REJECTED
    assert not any(m == "POST" for m, _ in fake.calls)

    pending = registry.execute("airflow_trigger_dag", {"conf": {"min_f1": 0.6}})
    assert pending.status is Status.CONFIRMATION_REQUIRED
    assert not any(m == "POST" for m, _ in fake.calls)  # nothing triggered yet
    done = registry.confirm(pending.data["confirmation_id"])
    assert done.status is Status.OK and done.data["triggered"] is True
    assert ("POST", "/api/v2/dags/churn_training_pipeline/dagRuns") in fake.calls


def test_airflow_outage_is_reported_as_unavailable(registry: ToolRegistry):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = airflow_tools.AirflowClient(registry.settings, transport=httpx.MockTransport(refuse))
    registry.register_all(airflow_tools.build_tools(registry.settings, client))
    res = registry.execute("airflow_dag_runs", {})
    assert res.status is Status.UNAVAILABLE and "unreachable" in res.error


def test_wrong_credentials_do_not_leak_the_password(registry: ToolRegistry):
    handler = lambda request: httpx.Response(401, json={"detail": "bad credentials"})  # noqa: E731
    client = airflow_tools.AirflowClient(registry.settings, transport=httpx.MockTransport(handler))
    registry.register_all(airflow_tools.build_tools(registry.settings, client))
    res = registry.execute("airflow_list_dags", {})
    assert res.status is Status.ERROR and "authentication failed" in res.error
    assert "pw-test-123456" not in registry.settings.log_file.read_text()
