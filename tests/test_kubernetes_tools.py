"""Kubernetes tools against a mocked API: namespace policy, confirmation, diagnosis, outages."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from urllib3.exceptions import HTTPError

from agent.schemas import Status
from agent.tools import kubernetes_tools
from agent.tools.base import ToolRegistry


@pytest.fixture
def k8s(registry: ToolRegistry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    return registry, core, apps


def make_pod(name="churn-api-1", phase="Running", ready=True, restarts=0, waiting=None):
    state = NS(running=None if waiting else NS(), waiting=NS(reason=waiting, message="") if waiting else None, terminated=None)
    cs = NS(name="churn-api", ready=ready, restart_count=restarts, state=state, last_state=NS(terminated=None), image="churn-api:1.0.0")
    return NS(
        metadata=NS(name=name, creation_timestamp=datetime.now(UTC)),
        status=NS(phase=phase, container_statuses=[cs], conditions=[]),
        spec=NS(node_name="n1", containers=[NS(readiness_probe=None, liveness_probe=None, resources=NS(requests={}, limits={}))]),
    )


def test_modification_in_unapproved_namespace_is_rejected_and_never_reaches_the_cluster(k8s):
    registry, _, apps = k8s
    scale = registry.execute("k8s_scale_deployment", {"namespace": "kube-system", "deployment": "coredns", "replicas": 0})
    restart = registry.execute("k8s_restart_deployment", {"namespace": "production", "deployment": "payments"})
    assert scale.status is Status.REJECTED and restart.status is Status.REJECTED
    assert "not approved" in scale.error
    apps.patch_namespaced_deployment_scale.assert_not_called()
    apps.patch_namespaced_deployment.assert_not_called()
    assert registry.pending == {}  # no confirmation was even offered


def test_scale_beyond_the_replica_limit_is_rejected(k8s):
    registry, _, apps = k8s
    res = registry.execute("k8s_scale_deployment", {"replicas": 50})
    assert res.status is Status.REJECTED and "maximum" in res.error
    apps.patch_namespaced_deployment_scale.assert_not_called()


def test_cluster_change_requires_explicit_confirmation_then_executes_in_approved_namespace(k8s):
    registry, _, apps = k8s
    req = registry.execute("k8s_restart_deployment", {"deployment": "churn-api"})
    assert req.status is Status.CONFIRMATION_REQUIRED
    apps.patch_namespaced_deployment.assert_not_called()  # nothing happened yet
    done = registry.confirm(req.data["confirmation_id"])
    assert done.status is Status.OK
    name, ns, body = apps.patch_namespaced_deployment.call_args.args
    assert (name, ns) == ("churn-api", "mlops-agent")
    assert "restartedAt" in str(body)


def test_reads_are_also_limited_to_approved_namespaces(k8s):
    registry, core, _ = k8s
    assert registry.execute("k8s_list_pods", {"namespace": "kube-system"}).status is Status.REJECTED
    core.list_namespaced_pod.assert_not_called()


def test_unreachable_cluster_is_reported_as_unavailable(k8s):
    registry, core, _ = k8s
    core.list_namespaced_pod.side_effect = HTTPError("connection refused")
    res = registry.execute("k8s_list_pods", {})
    assert res.status is Status.UNAVAILABLE and res.data is None


def test_list_pods_reports_restarts_and_waiting_reason(k8s):
    registry, core, _ = k8s
    core.list_namespaced_pod.return_value = NS(items=[make_pod(ready=False, restarts=4, waiting="CrashLoopBackOff")])
    row = registry.execute("k8s_list_pods", {}).data["pods"][0]
    assert row["restarts"] == 4 and row["waiting_reason"] == "CrashLoopBackOff" and row["ready"] == "0/1"


def test_diagnosis_flags_not_ready_pod_and_uses_logs_as_evidence(k8s):
    registry, core, apps = k8s
    apps.read_namespaced_deployment.return_value = NS(
        spec=NS(replicas=2, selector=NS(match_labels={"app": "churn-api"})), status=NS(ready_replicas=0))
    core.list_namespaced_pod.return_value = NS(items=[make_pod("a", ready=False), make_pod("b", ready=False)])
    core.read_namespaced_pod_log.return_value = "ERROR churn-api MODEL LOAD FAILED path=/bad/model.joblib FileNotFoundError"
    core.list_namespaced_event.return_value = NS(items=[NS(
        type="Warning", reason="Unhealthy", message="Readiness probe failed: HTTP probe failed with statuscode: 503",
        count=9, last_timestamp=datetime.now(UTC), event_time=None, metadata=NS(creation_timestamp=datetime.now(UTC)),
        involved_object=NS(kind="Pod", name="a"))])
    data = registry.execute("k8s_diagnose_deployment", {}).data
    assert data["healthy"] is False and data["ready"] == 0 and len(data["unhealthy_pods"]) == 2
    causes = " | ".join(data["likely_causes"])
    assert "readiness probe is failing" in causes and "model file could not be loaded" in causes
    assert data["warning_events"][0]["reason"] == "Unhealthy"


def test_pod_logs_are_redacted_before_reaching_the_agent(k8s):
    registry, core, _ = k8s
    core.read_namespaced_pod_log.return_value = "starting\nAUTH password=hunter2 Bearer abc.def.ghi"
    lines = registry.execute("k8s_pod_logs", {"pod": "p"}).data["lines"]
    assert "hunter2" not in " ".join(lines) and "abc.def.ghi" not in " ".join(lines)
