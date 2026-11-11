"""Deterministic, evidence-based retraining policy.

Retraining is recommended ONLY when at least one explicit trigger is backed by tool evidence:
  * freshness policy: deployed model older than ``max_model_age_days``;
  * data freshness policy: deployed model trained on an older DVC dataset version than the current one;
  * failed quality gate in the latest Airflow training run;
  * model-degradation evidence supplied by the user.

A strong metric, an unhealthy deployment, or the mere existence of a better run is NOT a trigger.
The function is pure (no I/O) so the rules are unit-testable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Evidence:
    """What the correlator managed to collect; ``None`` means the tool was unavailable / not answered."""

    deployment: dict[str, Any] | None = None  # k8s_app_version data
    deployment_health: dict[str, Any] | None = None  # k8s_diagnose_deployment data
    deployed_run: dict[str, Any] | None = None  # mlflow_get_run data for the deployed model
    best_run: dict[str, Any] | None = None  # mlflow_best_run data
    current_dataset: dict[str, Any] | None = None  # dvc_dataset_version data
    airflow_runs: dict[str, Any] | None = None  # airflow_dag_runs data
    airflow_failed: dict[str, Any] | None = None  # airflow_failed_tasks data
    degradation_report: str | None = None
    errors: dict[str, str] = field(default_factory=dict)  # tool -> reason it produced nothing


def _age_days(started: str | None, now: datetime) -> float | None:
    if not started:
        return None
    return (now - datetime.fromisoformat(started)).total_seconds() / 86400


def assess(ev: Evidence, *, max_model_age_days: int, require_current_dataset: bool = True, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    facts: list[str] = []
    uncertainties: list[str] = []
    triggers: list[str] = []

    # --- deployment ------------------------------------------------------------------------
    cfg = (ev.deployment or {}).get("config", {})
    deployed_dataset, deployed_run_id = cfg.get("DATASET_VERSION"), cfg.get("MODEL_RUN_ID")
    if ev.deployment:
        facts.append(f"Deployed app {ev.deployment['version_label']} (image {ev.deployment['image']}) serves model run "
                     f"{deployed_run_id} trained on dataset {deployed_dataset}.")
    else:
        uncertainties.append("Deployment details unknown: " + ev.errors.get("k8s_app_version", "Kubernetes evidence missing") + ".")
    if ev.deployment_health:
        h = ev.deployment_health
        facts.append(f"Deployment health: {h['ready']}/{h['desired']} replicas ready (healthy={h['healthy']}).")
    else:
        uncertainties.append("Deployment health unknown: " + ev.errors.get("k8s_diagnose_deployment", "no health evidence") + ".")

    # --- model freshness -------------------------------------------------------------------
    age = None
    if ev.deployed_run:
        age = _age_days(ev.deployed_run.get("started"), now)
        f1 = ev.deployed_run.get("metrics", {}).get("f1")
        facts.append(f"Deployed model run '{ev.deployed_run.get('name')}' has F1={f1} and was trained {age:.1f} days ago."
                     if age is not None else f"Deployed model run '{ev.deployed_run.get('name')}' has F1={f1}.")
        recorded = ev.deployed_run.get("provenance", {}).get("dvc.dataset_version")
        if recorded and deployed_dataset and recorded != deployed_dataset:
            uncertainties.append(f"ConfigMap says dataset {deployed_dataset} but MLflow run recorded {recorded}: provenance mismatch.")
    else:
        uncertainties.append("Deployed model's MLflow run could not be inspected: " + ev.errors.get("mlflow_get_run", "run id unknown") + ".")
    if age is not None and age > max_model_age_days:
        triggers.append(f"Freshness policy: deployed model is {age:.0f} days old (max {max_model_age_days}).")

    # --- dataset freshness -----------------------------------------------------------------
    current = (ev.current_dataset or {}).get("dataset_version")
    if ev.current_dataset:
        facts.append(f"Current DVC dataset version is {current} (md5 {str(ev.current_dataset.get('md5'))[:8]}...).")
    else:
        uncertainties.append("Current dataset version unknown: " + ev.errors.get("dvc_dataset_version", "DVC evidence missing") + ".")
    if require_current_dataset and current and deployed_dataset and current != deployed_dataset:
        triggers.append(f"Data freshness policy: deployed model was trained on dataset {deployed_dataset}, current is {current}.")

    # --- best candidate --------------------------------------------------------------------
    if ev.best_run:
        b = ev.best_run["best"]
        facts.append(f"Best recent run by {ev.best_run['metric']}: '{b['name']}' ({ev.best_run['metric']}="
                     f"{b['metrics'].get(ev.best_run['metric'])}, dataset {b['dataset_version']}).")
        for caveat in ev.best_run.get("caveats", []):
            uncertainties.append(f"Best-run caveat: {caveat}.")
    else:
        uncertainties.append("Best recent run unknown: " + ev.errors.get("mlflow_best_run", "no MLflow evidence") + ".")

    # --- airflow ---------------------------------------------------------------------------
    gate_failed = False
    if ev.airflow_runs:
        runs = ev.airflow_runs.get("runs", [])
        states = [r["state"] for r in runs]
        facts.append(f"Recent Airflow runs (newest first): {', '.join(states) or 'none'}.")
        if ev.airflow_failed and ev.airflow_failed.get("failed_run"):
            root = ev.airflow_failed.get("root_cause_task")
            latest_failed = ev.airflow_failed["failed_run"]["dag_run_id"]
            if runs and runs[0]["dag_run_id"] == latest_failed:
                facts.append(f"Latest run failed at task '{root}'.")
                gate_failed = root == "quality_gate"
    else:
        uncertainties.append("Airflow run history unknown: " + ev.errors.get("airflow_dag_runs", "no Airflow evidence") + ".")
    if gate_failed:
        triggers.append("Latest training run failed its quality gate.")

    # --- degradation -----------------------------------------------------------------------
    if ev.degradation_report:
        triggers.append(f"User-supplied degradation evidence: {ev.degradation_report[:200]}")

    # --- verdict ---------------------------------------------------------------------------
    enough = ev.deployment is not None and ev.current_dataset is not None and ev.deployed_run is not None
    if triggers:
        verdict = "retraining_recommended"
        actions = ["Review the trigger(s) above; retraining is a recommendation only and has NOT been started.",
                   "If you approve, ask me to trigger the training DAG - it will require your explicit /confirm."]
    elif enough:
        verdict = "retraining_not_required"
        actions = ["No retraining trigger is met; keep the current deployment and re-check after new data or a failed gate."]
    else:
        verdict = "insufficient_evidence"
        actions = ["Restore the unavailable tool(s) listed under uncertainties, then re-run this assessment."]
    return {"verdict": verdict, "triggers_met": triggers, "facts": facts, "uncertainties": uncertainties,
            "recommended_actions": actions, "retraining_started": False,
            "policy": {"max_model_age_days": max_model_age_days, "require_current_dataset": require_current_dataset}}
