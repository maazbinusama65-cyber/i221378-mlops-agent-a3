"""Training validation/quality gate, dataset provenance, and Kubernetes manifest requirements."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.make_dataset import generate
from training import core
from training.provenance import dataset_provenance, file_md5

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- data validation (DAG stage 1)
def test_valid_dataset_passes_validation(tmp_path):
    path = tmp_path / "ok.csv"
    generate("v1").to_csv(path, index=False)
    assert core.validate_dataset(path)["rows"] == 1500


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.drop(columns=["usage_hours"]), "missing columns"),
    (lambda d: d.head(50), "too few rows"),
    (lambda d: d.assign(monthly_charges=d["monthly_charges"].where(d.index % 7 != 0)), "null values"),
    (lambda d: d.assign(churn=d["churn"] + 2), "binary"),
    (lambda d: d.assign(churn=0), "class balance"),
])
def test_corrupted_datasets_fail_validation_with_a_clear_reason(tmp_path, mutate, message):
    path = tmp_path / "bad.csv"
    mutate(generate("v1")).to_csv(path, index=False)
    with pytest.raises(core.ValidationError, match=message):
        core.validate_dataset(path)


def test_missing_dataset_file_fails_validation(tmp_path):
    with pytest.raises(core.ValidationError, match="not found"):
        core.validate_dataset(tmp_path / "nope.csv")


# ---------------------------------------------------------------- quality gate (DAG stage 5)
def test_quality_gate_blocks_weak_or_overfit_models():
    good = {"f1": 0.8, "recall": 0.7, "train_f1": 0.85, "test_f1": 0.8}
    assert core.check_quality_gate(good, core.QualityGate()) == []
    weak = core.check_quality_gate({"f1": 0.3, "recall": 0.2, "train_f1": 0.35, "test_f1": 0.3}, core.QualityGate())
    assert len(weak) == 2 and "min_f1" in weak[0]
    overfit = core.check_quality_gate({"f1": 0.7, "recall": 0.7, "train_f1": 1.0, "test_f1": 0.7}, core.QualityGate())
    assert any("gap" in v for v in overfit)


def test_training_pipeline_end_to_end_produces_metrics(tmp_path):
    path = tmp_path / "d.csv"
    generate("v2").to_csv(path, index=False)
    splits = core.preprocess(path, tmp_path / "out")
    _, metrics = core.train_and_evaluate(splits, "logistic_regression", {"C": 1.0})
    assert 0.5 < metrics["f1"] <= 1 and metrics["test_rows"] == 750


# ---------------------------------------------------------------- provenance
def test_dataset_provenance_records_version_md5_and_dirty_flag(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    md5 = file_md5(data)
    (tmp_path / "data.csv.dvc").write_text(yaml.safe_dump({"outs": [{"md5": md5, "path": "data.csv"}], "meta": {"version": "v9"}}))
    clean = dataset_provenance(data, tmp_path)
    assert clean["dvc.dataset_version"] == "v9" and clean["dvc.dirty"] == "false" and clean["dvc.md5"] == md5
    data.write_text("a\n1\n2\n")
    assert dataset_provenance(data, tmp_path)["dvc.dirty"] == "true"


def test_dataset_generation_is_deterministic():
    pd.testing.assert_frame_equal(generate("v1"), generate("v1"))
    assert len(generate("v2")) == 3000


# ---------------------------------------------------------------- Kubernetes manifests
def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "kubernetes" / name).read_text())


def test_deployment_meets_assignment_requirements():
    dep = load("deployment.yaml")
    assert dep["spec"]["replicas"] >= 2
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= container["resources"][kind].keys()
    assert container["envFrom"][0]["configMapRef"]["name"] == load("configmap.yaml")["metadata"]["name"]


def test_service_is_clusterip_and_selects_the_deployment_pods():
    svc, dep = load("service.yaml"), load("deployment.yaml")
    assert svc["spec"]["type"] == "ClusterIP"
    assert svc["spec"]["selector"].items() <= dep["spec"]["template"]["metadata"]["labels"].items()


def test_configmap_holds_no_secrets():
    data = load("configmap.yaml")["data"]
    assert not any(word in key.lower() for key in data for word in ("password", "secret", "token", "key"))
