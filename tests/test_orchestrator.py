"""Orchestrator: tool selection, multi-tool flow, grounding, and refusal to act without confirmation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agent.llm import LLMError, LLMResponse, ToolCall
from agent.orchestrator import Orchestrator, check_citations
from agent.schemas import DvcNoArgs, Status
from agent.tools import kubernetes_tools
from agent.tools.base import Tool, ToolRegistry, ToolUnavailable
from tests.conftest import ScriptedLLM, call, say


def tool_messages(llm: ScriptedLLM) -> list[dict]:
    return [m for m in llm.seen[-1] if m["role"] == "tool"]


def test_agent_selects_and_executes_the_requested_tool(registry: ToolRegistry):
    registry.register(Tool("dvc_dataset_version", "dvc", "v", DvcNoArgs, lambda a: {"dataset_version": "v2", "md5": "abc"}))
    llm = ScriptedLLM([call("dvc_dataset_version"), say("Known facts: dataset v2 [E1]")])
    resp = Orchestrator(llm, registry).run("Which dataset version is tracked?")
    assert [r.tool for r in resp.evidence] == ["dvc_dataset_version"] and resp.evidence[0].ok
    assert resp.steps == 2 and "v2" in resp.answer
    assert json.loads(tool_messages(llm)[0]["content"])["data"]["dataset_version"] == "v2"  # LLM saw the real tool output


def test_multi_tool_request_chains_results_across_modules(registry: ToolRegistry):
    registry.register(Tool("mlflow_best_run", "mlflow", "b", DvcNoArgs, lambda a: {"best": {"run_id": "r9", "dataset_version": "v2"}}))
    registry.register(Tool("dvc_resolve_reference", "dvc", "r", DvcNoArgs, lambda a: {"found": True, "version": "v2"}))
    registry.register(Tool("k8s_diagnose_deployment", "kubernetes", "k", DvcNoArgs, lambda a: {"healthy": True}))
    script = [
        LLMResponse(tool_calls=[ToolCall("1", "mlflow_best_run", {}, "{}")]),
        LLMResponse(tool_calls=[ToolCall("2", "dvc_resolve_reference", {}, "{}"), ToolCall("3", "k8s_diagnose_deployment", {}, "{}")]),
        say("done [E1][E2][E3]"),
    ]
    resp = Orchestrator(ScriptedLLM(script), registry).run("best model, dataset, deployment?")
    assert [r.tool for r in resp.evidence] == ["mlflow_best_run", "dvc_resolve_reference", "k8s_diagnose_deployment"]
    assert len({r.request_id for r in resp.evidence}) == 1  # one request id correlates every tool call


def test_agent_does_not_invent_results_when_a_tool_is_unavailable(registry: ToolRegistry):
    def down(_):
        raise ToolUnavailable("Kubernetes API unreachable")

    registry.register(Tool("k8s_diagnose_deployment", "kubernetes", "k", DvcNoArgs, down))
    llm = ScriptedLLM([call("k8s_diagnose_deployment"), say("Known facts: none. Uncertainties: cluster state UNKNOWN [E1].")])
    resp = Orchestrator(llm, registry).run("Is my prediction API healthy?")
    shown = json.loads(tool_messages(llm)[0]["content"])
    assert shown["status"] == "unavailable" and "data" not in shown and "unreachable" in shown["error"]
    assert resp.evidence[0].status is Status.UNAVAILABLE and resp.evidence[0].data is None


def test_correlator_reports_insufficient_evidence_when_every_system_is_down(settings):
    import dataclasses

    from agent.orchestrator import build_registry

    dead = dataclasses.replace(
        settings, mlflow_uri="http://127.0.0.1:9", airflow_url="http://127.0.0.1:9",
        kubeconfig=str(settings.repo_root / "missing-kubeconfig"), tool_timeout_s=8.0,
    )
    registry = build_registry(dead)  # repo_root is an empty tmp dir, so DVC/git fail too: nothing may be invented
    res = registry.execute("assess_retraining_need", {})
    assert res.status is Status.OK
    assert res.data["verdict"] == "insufficient_evidence" and res.data["triggers_met"] == [] and res.data["facts"] == []
    assert res.data["retraining_started"] is False and res.data["uncertainties"]


def test_citations_to_nonexistent_evidence_are_flagged(registry: ToolRegistry):
    registry.register(Tool("t", "x", "d", DvcNoArgs, lambda a: {}))
    llm = ScriptedLLM([call("t"), say("Accuracy is 99% [E7] and ok [E1]")])
    resp = Orchestrator(llm, registry).run("q")
    assert "GROUNDING WARNING" in resp.answer and "E7" in resp.answer
    assert check_citations("fine [E1, E2] 【E1】", registry.ledger) == ["E2"]


def test_unauthorized_cluster_modification_via_the_agent_is_rejected(registry: ToolRegistry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    llm = ScriptedLLM([call("k8s_scale_deployment", {"namespace": "kube-system", "deployment": "coredns", "replicas": 0}),
                       say("The request was rejected [E1].")])
    resp = Orchestrator(llm, registry).run("scale coredns in kube-system to zero")
    assert resp.evidence[0].status is Status.REJECTED and resp.pending_confirmations == []
    apps.patch_namespaced_deployment_scale.assert_not_called()


def test_the_llm_cannot_confirm_its_own_write_request(registry: ToolRegistry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    llm = ScriptedLLM([
        call("k8s_restart_deployment", {}),
        call("confirm", {"confirmation_id": "anything"}, "c2"),  # no such tool exists for the model
        say("Waiting for the user to run /confirm."),
    ])
    resp = Orchestrator(llm, registry).run("restart the api")
    statuses = [r.status for r in resp.evidence]
    assert statuses == [Status.CONFIRMATION_REQUIRED, Status.UNKNOWN_TOOL]
    assert len(resp.pending_confirmations) == 1
    apps.patch_namespaced_deployment.assert_not_called()


def test_malformed_tool_arguments_from_the_llm_become_an_invalid_request(registry: ToolRegistry):
    registry.register(Tool("dvc_dataset_version", "dvc", "v", DvcNoArgs, lambda a: {}))
    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall("1", "dvc_dataset_version", None, "{not json")]), say("could not run it")])
    resp = Orchestrator(llm, registry).run("q")
    assert resp.evidence[0].status is Status.INVALID_REQUEST


def test_llm_outage_returns_a_clear_message_without_conclusions(registry: ToolRegistry):
    class Dead:
        def chat(self, messages, tools):
            raise LLMError("rate limited")

    resp = Orchestrator(Dead(), registry).run("anything")
    assert resp.error and "unavailable" in resp.answer and resp.evidence == []


def test_runaway_tool_loops_are_stopped(registry: ToolRegistry):
    registry.register(Tool("t", "x", "d", DvcNoArgs, lambda a: {}))
    llm = ScriptedLLM([call("t") for _ in range(3)])
    resp = Orchestrator(llm, registry, max_steps=3).run("loop forever")
    assert resp.error == "max_steps" and len(resp.evidence) == 3
