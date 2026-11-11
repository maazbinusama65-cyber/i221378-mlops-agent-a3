"""Cross-module correlator: gathers evidence from all four systems and applies the retraining policy."""

from __future__ import annotations

from typing import Any

from agent import policy
from agent.config import Settings
from agent.schemas import RetrainArgs, Status
from agent.tools.base import Tool, ToolRegistry


def build_tools(settings: Settings, registry: ToolRegistry) -> list[Tool]:
    def gather(name: str, args: dict[str, Any], ev: policy.Evidence) -> dict[str, Any] | None:
        res = registry.execute(name, args)
        if res.status is Status.OK:
            return res.data
        ev.errors[name] = f"{res.status.value}: {res.error}"
        return None

    def assess_retraining_need(args: RetrainArgs) -> dict[str, Any]:
        ev = policy.Evidence(degradation_report=args.degradation_report)
        k8s_args = {"namespace": settings.k8s_namespaces[0], "deployment": settings.k8s_deployment}
        ev.deployment = gather("k8s_app_version", k8s_args, ev)
        ev.deployment_health = gather("k8s_diagnose_deployment", k8s_args, ev)
        ev.current_dataset = gather("dvc_dataset_version", {}, ev)
        ev.best_run = gather("mlflow_best_run", {"metric": "f1", "recent": 20}, ev)
        run_id = ((ev.deployment or {}).get("config") or {}).get("MODEL_RUN_ID")
        if run_id:
            ev.deployed_run = gather("mlflow_get_run", {"run_id": run_id}, ev)
        ev.airflow_runs = gather("airflow_dag_runs", {"limit": 5}, ev)
        if ev.airflow_runs and any(r["state"] == "failed" for r in ev.airflow_runs.get("runs", [])[:1]):
            ev.airflow_failed = gather("airflow_failed_tasks", {}, ev)
        return policy.assess(ev, max_model_age_days=settings.max_model_age_days,
                             require_current_dataset=settings.require_current_dataset)

    return [Tool("assess_retraining_need", "analysis",
                 "Correlate MLflow + DVC + Kubernetes + Airflow evidence and apply the explicit retraining policy. "
                 "Never starts retraining. Pass degradation_report ONLY if the user supplied one.",
                 RetrainArgs, assess_retraining_need, timeout_s=240.0)]
