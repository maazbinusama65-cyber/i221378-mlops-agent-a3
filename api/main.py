"""Churn inference service (the ML application the agent operates on Kubernetes).

Endpoints
  GET  /health   liveness  - process is up (always 200 while the event loop works)
  GET  /ready    readiness - 200 only when the model is loaded, else 503
  GET  /version  deployed application / model / dataset provenance (from ConfigMap env)
  POST /predict  churn probability for one customer

Configuration comes from environment variables (injected from a Kubernetes ConfigMap).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "contract_months", "usage_hours"]

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn-api")

state: dict = {"model": None, "loaded_at": None, "error": None}


def load_model() -> None:
    path = Path(os.environ.get("MODEL_PATH", "/app/models/model.joblib"))
    try:
        state["model"] = joblib.load(path)
        state["loaded_at"] = time.time()
        state["error"] = None
        log.info("model loaded from %s", path)
    except Exception as exc:  # noqa: BLE001 - surfaced through /ready and the logs
        state["model"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("MODEL LOAD FAILED path=%s error=%s", path, state["error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn inference service", lifespan=lifespan)


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    contract_months: int = Field(ge=1)
    usage_hours: float = Field(ge=0)


@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    if state["model"] is None:
        log.warning("readiness check failed: model not loaded (%s)", state["error"])
        raise HTTPException(status_code=503, detail=f"model not loaded: {state['error']}")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    return {
        "app_version": os.environ.get("APP_VERSION", "dev"),
        "model_run_id": os.environ.get("MODEL_RUN_ID", "unknown"),
        "dataset_version": os.environ.get("DATASET_VERSION", "unknown"),
        "model_loaded": state["model"] is not None,
    }


@app.post("/predict")
def predict(customer: Customer) -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(state["model"].predict_proba(frame)[0, 1])
    log.info("prediction served churn_probability=%.3f", proba)
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}
