"""Structured tool inputs/outputs shared by every module and the orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Status(StrEnum):
    OK = "ok"
    ERROR = "error"  # the tool ran but the operation failed
    UNAVAILABLE = "unavailable"  # the backing service could not be reached
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_TOOL = "unknown_tool"
    REJECTED = "rejected"  # blocked by a safety policy (namespace / allowlist)
    CONFIRMATION_REQUIRED = "confirmation_required"


class ToolResult(BaseModel):
    """Every tool returns this envelope - the agent only reasons over these records."""

    tool: str
    status: Status
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""
    evidence_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


class ToolArgs(BaseModel):
    """Base class for tool arguments: reject unexpected fields so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ---- DVC ---------------------------------------------------------------------------------
class DvcNoArgs(ToolArgs):
    pass


class DvcDatasetArgs(ToolArgs):
    path: str = Field("data/churn.csv", description="Dataset path tracked by DVC")


class DvcResolveArgs(ToolArgs):
    md5: str | None = Field(None, description="Dataset md5 recorded on an MLflow run (tag dvc.md5)")
    version: str | None = Field(None, description="Dataset version label, e.g. v1")
    path: str = "data/churn.csv"


class DvcCompareArgs(ToolArgs):
    rev_a: str = Field(description="Git revision or tag, e.g. data-v1")
    rev_b: str = Field(description="Git revision or tag, e.g. data-v2")
    path: str = "data/churn.csv"


class DvcReproduceArgs(ToolArgs):
    rev: str = Field(description="Git tag/commit of the dataset version to reproduce")
    path: str = "data/churn.csv"
    verify: bool = Field(False, description="Also export the dataset into a sandbox dir and verify its md5")


# ---- MLflow ------------------------------------------------------------------------------
class MlflowNoArgs(ToolArgs):
    pass


class MlflowListRunsArgs(ToolArgs):
    experiment: str | None = Field(None, description="Experiment name; omit for all experiments")
    limit: int = Field(10, ge=1, le=50)


class MlflowRunArgs(ToolArgs):
    run_id: str


class MlflowBestArgs(ToolArgs):
    metric: str = Field("f1", description="Metric to maximise, e.g. f1, recall, roc_auc")
    experiment: str | None = None
    recent: int = Field(10, ge=1, le=100, description="Only consider the N most recent runs")


class MlflowCompareRunsArgs(ToolArgs):
    run_ids: list[str] | None = Field(None, description="Specific run ids to compare")
    last_n: int | None = Field(None, ge=2, le=20, description="Or compare the last N runs")
    experiment: str | None = None


class MlflowCompareExperimentsArgs(ToolArgs):
    experiments: list[str] = Field(min_length=2, description="Two or more experiment names")
    metric: str = "f1"


class MlflowRegistryArgs(ToolArgs):
    name: str = "churn-classifier"


# ---- Airflow -----------------------------------------------------------------------------
class AirflowNoArgs(ToolArgs):
    pass


class AirflowRunsArgs(ToolArgs):
    dag_id: str = "churn_training_pipeline"
    limit: int = Field(5, ge=1, le=20)


class AirflowRunArgs(ToolArgs):
    dag_id: str = "churn_training_pipeline"
    run_id: str | None = Field(None, description="DAG run id; omit for the latest run")


class AirflowLogsArgs(ToolArgs):
    dag_id: str = "churn_training_pipeline"
    run_id: str | None = None
    task_id: str
    try_number: int | None = Field(None, ge=1)
    tail_lines: int = Field(40, ge=5, le=200)


class AirflowTriggerArgs(ToolArgs):
    dag_id: str = "churn_training_pipeline"
    conf: dict[str, Any] = Field(default_factory=dict, description="Optional DAG params, e.g. {'min_f1': 0.6}")


# ---- Kubernetes --------------------------------------------------------------------------
class K8sNsArgs(ToolArgs):
    namespace: str = "mlops-agent"


class K8sPodArgs(ToolArgs):
    pod: str
    namespace: str = "mlops-agent"


class K8sLogsArgs(ToolArgs):
    pod: str
    namespace: str = "mlops-agent"
    tail_lines: int = Field(40, ge=5, le=300)
    previous: bool = Field(False, description="Logs of the previous (crashed) container instance")


class K8sEventsArgs(ToolArgs):
    namespace: str = "mlops-agent"
    warnings_only: bool = True


class K8sDeploymentArgs(ToolArgs):
    deployment: str = "churn-api"
    namespace: str = "mlops-agent"


class K8sScaleArgs(ToolArgs):
    deployment: str = "churn-api"
    namespace: str = "mlops-agent"
    replicas: int = Field(ge=0)


# ---- Analysis ----------------------------------------------------------------------------
class RetrainArgs(ToolArgs):
    degradation_report: str | None = Field(
        None, description="Optional user-supplied evidence of model degradation (never invent this)"
    )
