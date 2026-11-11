"""MLflow tools against a fake MlflowClient."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from agent.schemas import Status
from agent.tools.base import ToolRegistry
from agent.tools.mlflow_tools import MlflowTools, build_tools


def make_run(rid, name, f1, *, train_f1=None, gate="passed", version="v2", dirty="false", rows=750, start=1_700_000_000_000):
    metrics = {"f1": f1, "recall": f1 - 0.05, "precision": f1, "train_f1": train_f1 if train_f1 is not None else f1 + 0.02, "test_rows": rows}
    tags = {"dvc.dataset_version": version, "dvc.md5": "a" * 32, "dvc.dirty": dirty, "git.commit": "c0ffee", "quality_gate": gate}
    return NS(info=NS(run_id=rid, run_name=name, experiment_id="1", status="FINISHED", start_time=start),
              data=NS(metrics=metrics, tags=tags, params={"algorithm": "random_forest"}))


class FakeClient:
    def __init__(self, runs, registered=()):
        self.runs, self.registered = runs, registered

    def search_experiments(self):
        return [NS(experiment_id="1", name="churn-training", last_update_time=1_700_000_000_000)]

    def get_experiment_by_name(self, name):
        return NS(experiment_id="1", name=name) if name == "churn-training" else None

    def search_runs(self, ids, filter_string=None, order_by=None, max_results=100):
        return self.runs[:max_results]

    def get_run(self, rid):
        return next(r for r in self.runs if r.info.run_id == rid)

    def search_model_versions(self, flt):
        return list(self.registered)

    def list_artifacts(self, rid):
        return [NS(path="model"), NS(path="metrics.json")]


@pytest.fixture
def mlflow(registry: ToolRegistry):
    tools = build_tools(registry.settings)
    registry.register_all(tools)
    holder = tools[0].fn.__self__  # the MlflowTools instance
    assert isinstance(holder, MlflowTools)
    return registry, holder


def test_best_run_picks_top_metric_and_explains_caveats_instead_of_claiming_readiness(mlflow):
    registry, tools = mlflow
    tools.client = FakeClient([
        make_run("r1", "overfit", 0.82, train_f1=1.0),
        make_run("r2", "balanced", 0.81, train_f1=0.84),
        make_run("r3", "weak", 0.60),
    ])
    data = registry.execute("mlflow_best_run", {"metric": "f1"}).data
    assert data["best"]["run_id"] == "r1" and [r["run_id"] for r in data["ranking"]] == ["r1", "r2", "r3"]
    text = " ".join(data["caveats"])
    assert "overfitting" in text and "within noise" in text and "not registered" in text
    assert data["production_ready"].startswith("not asserted")


def test_best_run_flags_failed_quality_gate_and_dirty_dataset(mlflow):
    registry, tools = mlflow
    tools.client = FakeClient([make_run("r1", "x", 0.9, gate="failed", dirty="true", version="untracked", rows=100)])
    text = " ".join(registry.execute("mlflow_best_run", {}).data["caveats"])
    assert "quality gate" in text and "differed from its recorded DVC version" in text and "no DVC dataset version" in text
    assert "small evaluation set" in text


def test_compare_runs_returns_winner_per_metric_and_requires_two_runs(mlflow):
    registry, tools = mlflow
    tools.client = FakeClient([make_run("r1", "a", 0.7), make_run("r2", "b", 0.8)])
    data = registry.execute("mlflow_compare_runs", {"last_n": 2}).data
    assert data["winner_per_metric"]["f1"]["run_id"] == "r2" and len(data["runs"]) == 2
    tools.client = FakeClient([make_run("r1", "a", 0.7)])
    assert registry.execute("mlflow_compare_runs", {"last_n": 2}).status is Status.ERROR


def test_get_run_exposes_dvc_and_git_provenance(mlflow):
    registry, tools = mlflow
    tools.client = FakeClient([make_run("r1", "a", 0.7, version="v1")], registered=[NS(name="churn-classifier", version="3")])
    data = registry.execute("mlflow_get_run", {"run_id": "r1"}).data
    assert data["provenance"]["dvc.dataset_version"] == "v1" and data["provenance"]["git.commit"] == "c0ffee"
    assert data["registered_versions"] == [{"name": "churn-classifier", "version": "3"}] and "model" in data["artifacts"]


def test_unknown_experiment_is_an_error_not_a_crash(mlflow):
    registry, tools = mlflow
    tools.client = FakeClient([])
    res = registry.execute("mlflow_list_runs", {"experiment": "nope"})
    assert res.status is Status.ERROR and "not found" in res.error


def test_mlflow_outage_is_reported_as_unavailable(mlflow):
    registry, tools = mlflow

    class Down(FakeClient):
        def search_runs(self, *a, **k):
            raise RequestsConnectionError("Max retries exceeded")

    tools.client = Down([])
    res = registry.execute("mlflow_best_run", {})
    assert res.status is Status.UNAVAILABLE and "unreachable" in res.error and res.data is None
