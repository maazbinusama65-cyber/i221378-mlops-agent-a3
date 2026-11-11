"""Runtime configuration, read from environment variables (never hard-coded credentials)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    # MLflow
    mlflow_uri: str = "http://127.0.0.1:5000"
    # Airflow 3 (JWT auth)
    airflow_url: str = "http://127.0.0.1:8080"
    airflow_user: str = "admin"
    airflow_password: str = field(default="", repr=False)
    approved_dags: tuple[str, ...] = ("churn_training_pipeline",)
    # Kubernetes
    k8s_namespaces: tuple[str, ...] = ("mlops-agent",)
    k8s_deployment: str = "churn-api"
    kubeconfig: str | None = None
    k8s_max_replicas: int = 5
    # DVC
    dvc_datasets: tuple[str, ...] = ("data/churn.csv",)
    # Safety / reliability
    tool_timeout_s: float = 30.0
    confirmation_ttl_s: float = 300.0
    log_file: Path = REPO_ROOT / "logs" / "agent.jsonl"
    # Retraining policy (explicit freshness policy)
    max_model_age_days: int = 30
    require_current_dataset: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)
        return cls(
            mlflow_uri=e.get("MLFLOW_TRACKING_URI", cls.mlflow_uri),
            airflow_url=e.get("AIRFLOW_BASE_URL", cls.airflow_url),
            airflow_user=e.get("AIRFLOW_USERNAME", cls.airflow_user),
            airflow_password=e.get("AIRFLOW_PASSWORD", ""),
            approved_dags=_csv(e.get("AGENT_APPROVED_DAGS", "churn_training_pipeline")),
            k8s_namespaces=_csv(e.get("AGENT_K8S_NAMESPACES", "mlops-agent")),
            k8s_deployment=e.get("AGENT_K8S_DEPLOYMENT", cls.k8s_deployment),
            kubeconfig=e.get("KUBECONFIG") or None,
            dvc_datasets=_csv(e.get("AGENT_DVC_DATASETS", "data/churn.csv")),
            tool_timeout_s=float(e.get("AGENT_TOOL_TIMEOUT_S", cls.tool_timeout_s)),
            log_file=Path(e.get("AGENT_LOG_FILE", str(cls.log_file))),
            max_model_age_days=int(e.get("AGENT_MAX_MODEL_AGE_DAYS", cls.max_model_age_days)),
        )
