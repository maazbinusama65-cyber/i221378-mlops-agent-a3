"""LLM client parsing/retry, the CLI, and the remaining read tools (MLflow registry, Kubernetes inspection)."""

from __future__ import annotations

import builtins
import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from agent import main as cli
from agent.config import Settings
from agent.llm import LLMError, OpenAICompatLLM, resolve_config
from agent.tools import kubernetes_tools
from agent.tools.mlflow_tools import MlflowTools, build_tools
from tests.conftest import ScriptedLLM, call, say
from tests.test_kubernetes_tools import make_pod
from tests.test_mlflow_tools import FakeClient, make_run


# ------------------------------------------------------------------------------------- LLM
def test_provider_autodetect_and_missing_credentials():
    cfg = resolve_config({"GROQ_API_KEY": "gsk_testtesttest"})
    assert cfg["provider"] == "groq" and "groq.com" in cfg["base_url"]
    with pytest.raises(LLMError):
        resolve_config({})
    with pytest.raises(LLMError):
        resolve_config({"LLM_PROVIDER": "openai"})
    assert resolve_config({"LLM_PROVIDER": "ollama", "LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "m"})["api_key"] == "not-needed"


def test_rate_limit_hint_is_honoured_and_response_parsed():
    assert OpenAICompatLLM._wait(Exception("Please try again in 2.5s."), 0) == 3.5
    assert OpenAICompatLLM._wait(Exception("nothing"), 2) == 8.0
    tool_call = NS(id="1", function=NS(name="dvc_dataset_version", arguments='{"path": "data/churn.csv"}'))
    bad_call = NS(id="2", function=NS(name="x", arguments="{oops"))
    resp = NS(choices=[NS(message=NS(content=None, tool_calls=[tool_call, bad_call]))], usage=NS(prompt_tokens=5, completion_tokens=2))
    parsed = OpenAICompatLLM._parse(resp)
    assert parsed.tool_calls[0].arguments == {"path": "data/churn.csv"} and parsed.tool_calls[1].arguments is None
    assert parsed.usage == {"prompt_tokens": 5, "completion_tokens": 2}


def test_llm_retries_rate_limits_then_succeeds_and_gives_up_eventually(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    llm = OpenAICompatLLM("key-123456", "m", "http://localhost:1/v1", max_retries=2)
    limited = openai.RateLimitError("try again in 1ms", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
    ok = NS(choices=[NS(message=NS(content="hi", tool_calls=None))], usage=None)
    llm._client = MagicMock()
    llm._client.chat.completions.create.side_effect = [limited, ok]
    assert llm.chat([], []).content == "hi"
    llm._client.chat.completions.create.side_effect = [limited, limited, limited]
    with pytest.raises(LLMError, match="unavailable after"):
        llm.chat([], [])


# ------------------------------------------------------------------------------------- CLI
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_LOG_FILE", str(tmp_path / "cli.jsonl"))
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "none"))

    def install(script):
        llm = ScriptedLLM(script)
        llm.model, llm.provider = "scripted", "test"
        monkeypatch.setattr(cli.OpenAICompatLLM, "from_env", classmethod(lambda cls, env=None: llm))
        return llm

    return install


def test_cli_single_question_saves_redacted_evidence_transcript(cli_env, tmp_path, capsys):
    cli_env([call("dvc_list_datasets"), say("Known facts: [E1]")])
    out = tmp_path / "t.json"
    assert cli.main(["--ask", "what datasets exist?", "--save", str(out)]) in (0, 1)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["question"] == "what datasets exist?" and saved["tool_evidence"][0]["tool"] == "dvc_list_datasets"
    assert "evidence" in capsys.readouterr().out


def test_cli_without_llm_credentials_exits_cleanly(monkeypatch, capsys):
    def boom(cls, env=None):
        raise LLMError("no key")

    monkeypatch.setattr(cli.OpenAICompatLLM, "from_env", classmethod(boom))
    assert cli.main(["--ask", "x"]) == 2 and "no key" in capsys.readouterr().err


def test_cli_interactive_confirm_flow(cli_env, monkeypatch, capsys):
    cli_env([call("k8s_restart_deployment", {}), say("Please run /confirm.")])
    answers = iter(["", "restart it", "/pending", "/confirm", "/confirm nope", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Pending confirmations" in out and "usage: /confirm <id>" in out and '"status": "rejected"' in out


# ------------------------------------------------------------------------------------- MLflow extras
def test_registry_listing_experiments_and_experiment_comparison(registry):
    tools = build_tools(registry.settings)
    registry.register_all(tools)
    holder: MlflowTools = tools[0].fn.__self__
    runs = [make_run("r1", "a", 0.7), make_run("r2", "b", 0.8)]

    class Client(FakeClient):
        def get_registered_model(self, name):
            return NS(aliases={"production": "1"})

        def search_model_versions(self, flt):
            return [NS(name="churn-classifier", version="1", run_id="r1")]

    holder.client = Client(runs)
    exps = registry.execute("mlflow_list_experiments", {}).data["experiments"]
    assert exps[0]["name"] == "churn-training" and exps[0]["runs"] == 2
    reg = registry.execute("mlflow_model_registry", {}).data
    assert reg["versions"][0]["aliases"] == ["production"] and reg["versions"][0]["f1"] == 0.7
    cmp = registry.execute("mlflow_compare_experiments", {"experiments": ["churn-training", "churn-training"]}).data
    assert cmp["experiments"][0]["best"] == 0.8 and cmp["experiments"][0]["runs"] == 2
    assert registry.execute("mlflow_list_runs", {"limit": 1}).data["count"] == 1


# ------------------------------------------------------------------------------------- Kubernetes extras
def test_service_events_version_and_pod_inspection(registry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    now = datetime.now(UTC)

    core.list_namespaced_service.return_value = NS(items=[NS(
        metadata=NS(name="churn-api"), spec=NS(type="ClusterIP", cluster_ip="10.0.0.1", selector={"app": "churn-api"},
                                               ports=[NS(port=80, target_port="http")]))])
    core.read_namespaced_endpoints.return_value = NS(subsets=[NS(addresses=[1, 2], not_ready_addresses=[3])])
    svc = registry.execute("k8s_services", {}).data["services"][0]
    assert svc["ready_endpoints"] == 2 and svc["not_ready_endpoints"] == 1 and svc["type"] == "ClusterIP"

    core.list_namespaced_event.return_value = NS(items=[
        NS(type="Warning", reason="BackOff", message="Back-off restarting", count=5, last_timestamp=now, event_time=None,
           metadata=NS(creation_timestamp=now), involved_object=NS(kind="Pod", name="p")),
        NS(type="Normal", reason="Pulled", message="ok", count=1, last_timestamp=now, event_time=None,
           metadata=NS(creation_timestamp=now), involved_object=NS(kind="Pod", name="p"))])
    assert [e["reason"] for e in registry.execute("k8s_events", {}).data["events"]] == ["BackOff"]
    assert len(registry.execute("k8s_events", {"warnings_only": False}).data["events"]) == 2

    container = NS(image="churn-api:2.0.0", env_from=[NS(config_map_ref=NS(name="cfg"))])
    apps.read_namespaced_deployment.return_value = NS(
        metadata=NS(labels={"app.kubernetes.io/version": "2.0.0"}, generation=3), spec=NS(replicas=2, template=NS(spec=NS(containers=[container]))),
        status=NS(observed_generation=3, ready_replicas=2))
    core.read_namespaced_config_map.return_value = NS(data={"APP_VERSION": "2.0.0", "DATASET_VERSION": "v2", "MODEL_RUN_ID": "r5"})
    ver = registry.execute("k8s_app_version", {}).data
    assert ver["image"] == "churn-api:2.0.0" and ver["config"]["DATASET_VERSION"] == "v2" and ver["replicas"]["ready"] == 2

    core.read_namespaced_pod.return_value = make_pod("p1", phase="Running", ready=False, restarts=2, waiting="CrashLoopBackOff")
    status = registry.execute("k8s_pod_status", {"pod": "p1"}).data
    assert status["containers"][0]["waiting_reason"] == "CrashLoopBackOff" and status["containers"][0]["restart_count"] == 2


def test_scale_executes_only_after_confirmation(registry):
    core, apps = MagicMock(), MagicMock()
    registry.register_all(kubernetes_tools.build_tools(registry.settings, core, apps))
    pending = registry.execute("k8s_scale_deployment", {"replicas": 3})
    assert pending.status.value == "confirmation_required" and "3 replicas" in pending.data["action"]
    assert registry.confirm(pending.data["confirmation_id"]).data["replicas"] == 3
    assert apps.patch_namespaced_deployment_scale.call_args.args[:2] == ("churn-api", "mlops-agent")


def test_settings_from_env_parses_lists_and_defaults():
    s = Settings.from_env({"AGENT_K8S_NAMESPACES": "a, b", "AIRFLOW_PASSWORD": "pw", "AGENT_MAX_MODEL_AGE_DAYS": "7"})
    assert s.k8s_namespaces == ("a", "b") and s.airflow_password == "pw" and s.max_model_age_days == 7
    assert "pw" not in repr(s)  # secrets never appear in repr/log output
