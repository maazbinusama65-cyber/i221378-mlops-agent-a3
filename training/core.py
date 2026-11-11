"""Training steps shared by ``training/train.py`` (CLI) and the Airflow DAG.

Pipeline: validate -> preprocess -> train -> evaluate (+ MLflow) -> quality gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training.provenance import dataset_provenance

log = logging.getLogger("training")

FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "contract_months", "usage_hours"]
TARGET = "churn"
REGISTERED_MODEL = "churn-classifier"
DEFAULT_EXPERIMENT = "churn-training"
MIN_ROWS = 500


class ValidationError(ValueError):
    """The input dataset failed a data-quality check."""


class QualityGateError(RuntimeError):
    """The evaluated model did not meet the promotion thresholds."""


@dataclass(frozen=True)
class QualityGate:
    min_f1: float = 0.55
    min_recall: float = 0.45
    max_overfit_gap: float = 0.25


def validate_dataset(path: Path) -> dict:
    """Schema and data-quality checks. Raises ValidationError on any violation."""
    if not path.exists():
        raise ValidationError(f"dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValidationError(f"missing columns: {missing}")
    if len(df) < MIN_ROWS:
        raise ValidationError(f"too few rows: {len(df)} < {MIN_ROWS}")
    nulls = int(df[[*FEATURES, TARGET]].isna().sum().sum())
    if nulls:
        raise ValidationError(f"{nulls} null values in feature/target columns")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValidationError("target column must be binary (0/1)")
    rate = float(df[TARGET].mean())
    if not 0.02 <= rate <= 0.98:
        raise ValidationError(f"degenerate class balance (positive rate {rate:.3f})")
    report = {"rows": int(len(df)), "positive_rate": round(rate, 4)}
    log.info("validation passed: %s", report)
    return report


def preprocess(path: Path, out_dir: Path, seed: int = 42) -> dict[str, str]:
    """Split into train/test CSVs (stratified)."""
    df = pd.read_csv(path)
    train, test = train_test_split(df, test_size=0.25, random_state=seed, stratify=df[TARGET])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": out_dir / "train.csv", "test": out_dir / "test.csv"}
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    log.info("preprocessed: train=%d test=%d", len(train), len(test))
    return {k: v.as_posix() for k, v in paths.items()}


def build_model(algorithm: str, params: dict):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            class_weight=params.get("class_weight"),
            random_state=42,
        )
    if algorithm == "logistic_regression":
        return LogisticRegression(C=float(params.get("C", 1.0)), class_weight=params.get("class_weight"), max_iter=500)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def train_and_evaluate(splits: dict[str, str], algorithm: str, params: dict):
    train = pd.read_csv(splits["train"])
    test = pd.read_csv(splits["test"])
    model = build_model(algorithm, params)
    model.fit(train[FEATURES], train[TARGET])
    metrics: dict[str, float] = {}
    for name, frame in (("train", train), ("test", test)):
        pred = model.predict(frame[FEATURES])
        metrics[f"{name}_f1"] = float(f1_score(frame[TARGET], pred))
    pred = model.predict(test[FEATURES])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    metrics.update(
        f1=metrics["test_f1"],
        precision=float(precision_score(test[TARGET], pred, zero_division=0)),
        recall=float(recall_score(test[TARGET], pred)),
        accuracy=float(accuracy_score(test[TARGET], pred)),
        roc_auc=float(roc_auc_score(test[TARGET], proba)),
        test_rows=float(len(test)),
    )
    return model, {k: round(v, 4) for k, v in metrics.items()}


def check_quality_gate(metrics: dict[str, float], gate: QualityGate) -> list[str]:
    """Return the list of violated criteria (empty list == gate passed)."""
    problems = []
    if metrics["f1"] < gate.min_f1:
        problems.append(f"f1 {metrics['f1']:.3f} < min_f1 {gate.min_f1}")
    if metrics["recall"] < gate.min_recall:
        problems.append(f"recall {metrics['recall']:.3f} < min_recall {gate.min_recall}")
    gap = metrics["train_f1"] - metrics["test_f1"]
    if gap > gate.max_overfit_gap:
        problems.append(f"train/test F1 gap {gap:.3f} > {gate.max_overfit_gap}")
    return problems


def log_run(
    model,
    metrics: dict[str, float],
    *,
    algorithm: str,
    params: dict,
    data_path: Path,
    repo_root: Path,
    experiment: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
    register: bool = False,
    gate_result: list[str] | None = None,
) -> str:
    """Log params/metrics/tags/model to MLflow and return the run id."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"algorithm": algorithm, **{k: str(v) for k, v in params.items()}})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(dataset_provenance(data_path, repo_root))
        if extra_tags:
            mlflow.set_tags(extra_tags)
        if gate_result is not None:
            mlflow.set_tag("quality_gate", "passed" if not gate_result else "failed")
            mlflow.log_dict({"violations": gate_result}, "quality_gate.json")
        mlflow.log_text(json.dumps(metrics, indent=2), "metrics.json")
        mlflow.sklearn.log_model(model, name="model", registered_model_name=REGISTERED_MODEL if register else None)
        return run.info.run_id


def set_candidate_alias(run_id: str) -> str | None:
    """Mark the model version produced by ``run_id`` as the ``candidate`` alias (never production)."""
    client = MlflowClient()
    for version in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        if version.run_id == run_id:
            client.set_registered_model_alias(REGISTERED_MODEL, "candidate", version.version)
            return str(version.version)
    return None
