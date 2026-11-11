"""The FastAPI inference service: liveness vs readiness semantics the Kubernetes probes rely on."""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api import main as api_main

PAYLOAD = {"tenure_months": 3, "monthly_charges": 95.0, "support_tickets": 4, "contract_months": 1, "usage_hours": 15.0}


@pytest.fixture
def healthy_client(tmp_path, monkeypatch):
    X = [[1, 50, 0, 12, 40], [2, 90, 5, 1, 10], [60, 40, 0, 24, 50], [3, 95, 4, 1, 12]] * 5
    y = [0, 1, 0, 1] * 5
    path = tmp_path / "model.joblib"
    joblib.dump(LogisticRegression(max_iter=500).fit(X, y), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("DATASET_VERSION", "v7")
    with TestClient(api_main.app) as client:
        yield client


def test_ready_and_predict_work_when_model_is_loaded(healthy_client):
    assert healthy_client.get("/health").status_code == 200
    assert healthy_client.get("/ready").status_code == 200
    body = healthy_client.post("/predict", json=PAYLOAD).json()
    assert 0 <= body["churn_probability"] <= 1 and isinstance(body["churn"], bool)


def test_version_endpoint_reports_configmap_provenance(healthy_client):
    assert healthy_client.get("/version").json() | {} == {
        "app_version": "9.9.9", "model_run_id": "unknown", "dataset_version": "v7", "model_loaded": True}


def test_missing_model_keeps_liveness_green_but_readiness_red(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does-not-exist.joblib"))
    with TestClient(api_main.app) as client:
        assert client.get("/health").status_code == 200  # process is alive: no restart loop
        ready = client.get("/ready")
        assert ready.status_code == 503 and "model not loaded" in ready.json()["detail"]
        assert client.post("/predict", json=PAYLOAD).status_code == 503


def test_prediction_input_is_validated(healthy_client):
    assert healthy_client.post("/predict", json={**PAYLOAD, "tenure_months": -5}).status_code == 422
