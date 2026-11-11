"""Churn training pipeline (Airflow 3).

validate_data -> preprocess -> train_model -> evaluate_and_log (MLflow) -> quality_gate -> register_candidate

* Every task logs through the ``airflow.task`` logger so the agent can fetch diagnostics.
* Transient failures retry (``retries=2``); data-validation and quality-gate failures raise
  ``AirflowFailException`` so they fail immediately and are never retried.
* ``register_candidate`` only runs when ``quality_gate`` succeeded, so a failed gate can never promote
  a model. Even on success the model only receives the ``candidate`` alias - production promotion is manual.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

try:  # Airflow 3.x moved the exception around between minors
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # pragma: no cover
    from airflow.exceptions import AirflowFailException

PROJECT_ROOT = Path("/opt/airflow/project")
log = logging.getLogger("airflow.task")

DEFAULT_ARGS = {
    "owner": "mlops-agent",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


def _params() -> dict:
    return get_current_context()["params"]


@dag(
    dag_id="churn_training_pipeline",
    description="Validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> candidate registration",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training", "approved"],
    params={
        "data_path": Param("data/churn.csv", type="string", description="Dataset path relative to repo root"),
        "algorithm": Param("random_forest", enum=["random_forest", "logistic_regression"]),
        "n_estimators": Param(200, type="integer"),
        "max_depth": Param(8, type=["integer", "null"]),
        "min_f1": Param(0.55, type="number", description="Quality-gate minimum F1"),
        "min_recall": Param(0.45, type="number"),
    },
)
def churn_training_pipeline():
    @task
    def validate_data() -> dict:
        from training import core

        data = PROJECT_ROOT / _params()["data_path"]
        try:
            report = core.validate_dataset(data)
        except core.ValidationError as exc:
            log.error("DATA VALIDATION FAILED: %s", exc)
            raise AirflowFailException(f"data validation failed: {exc}") from exc
        log.info("validated %s: %s", data, report)
        return {"data_path": str(data), **report}

    @task
    def preprocess(validated: dict) -> dict:
        from training import core

        splits = core.preprocess(Path(validated["data_path"]), PROJECT_ROOT / "data" / "processed")
        log.info("splits written: %s", splits)
        return {**validated, **splits}

    @task
    def train_model(prepared: dict) -> dict:
        import joblib

        from training import core

        p = _params()
        params = {"n_estimators": p["n_estimators"], "max_depth": p["max_depth"]}
        params = {k: v for k, v in params.items() if v is not None}
        model, metrics = core.train_and_evaluate(prepared, p["algorithm"], params)
        model_path = PROJECT_ROOT / "data" / "processed" / "airflow_model.joblib"
        joblib.dump(model, model_path)
        log.info("trained %s params=%s metrics=%s", p["algorithm"], params, metrics)
        return {**prepared, "metrics": metrics, "model_path": str(model_path), "model_params": params}

    @task
    def evaluate_and_log(trained: dict) -> dict:
        import joblib

        from training import core

        ctx = get_current_context()
        p = ctx["params"]
        gate = core.QualityGate(min_f1=float(p["min_f1"]), min_recall=float(p["min_recall"]))
        violations = core.check_quality_gate(trained["metrics"], gate)
        model = joblib.load(trained["model_path"])
        run_id = core.log_run(
            model,
            trained["metrics"],
            algorithm=p["algorithm"],
            params=trained["model_params"],
            data_path=Path(trained["data_path"]),
            repo_root=PROJECT_ROOT,
            run_name=f"airflow-{ctx['run_id']}",
            extra_tags={"trigger": "airflow", "airflow.dag_run_id": ctx["run_id"]},
            gate_result=violations,
        )
        log.info("logged MLflow run %s (gate violations: %s)", run_id, violations)
        return {"run_id": run_id, "metrics": trained["metrics"], "violations": violations}

    @task
    def quality_gate(evaluated: dict) -> str:
        if evaluated["violations"]:
            msg = "; ".join(evaluated["violations"])
            log.error("QUALITY GATE FAILED for run %s: %s", evaluated["run_id"], msg)
            raise AirflowFailException(f"quality gate failed: {msg}")
        log.info("quality gate passed for run %s", evaluated["run_id"])
        return evaluated["run_id"]

    @task
    def register_candidate(run_id: str) -> dict:
        import mlflow

        from training import core

        mv = mlflow.register_model(f"runs:/{run_id}/model", core.REGISTERED_MODEL)
        core.set_candidate_alias(run_id)
        log.info("registered %s v%s with alias 'candidate' (production promotion stays manual)", mv.name, mv.version)
        return {"model": mv.name, "version": str(mv.version), "alias": "candidate"}

    validated = validate_data()
    prepared = preprocess(validated)
    trained = train_model(prepared)
    evaluated = evaluate_and_log(trained)
    gated = quality_gate(evaluated)
    register_candidate(gated)


churn_training_pipeline()
