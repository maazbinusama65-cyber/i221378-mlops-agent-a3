#!/usr/bin/env bash
# One-shot environment bootstrap on a fresh clone (Docker Desktop required). Run from the repo root with the
# virtualenv activated:   python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

echo "[1/5] MLflow + Airflow"
docker compose up -d --build mlflow airflow
until curl -fsS "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; do sleep 3; done

echo "[2/5] DVC data objects"
bash scripts/bootstrap_data.sh

echo "[3/5] Experiments, registry aliases, serving model"
bash scripts/seed_experiments.sh
python scripts/set_registry_aliases.py
python scripts/sync_deployed_run_id.py
dvc add models/model.joblib

echo "[4/5] Kubernetes (k3s) + inference service"
bash scripts/k8s-up.sh

echo "[5/5] Done. Export before running the agent:"
echo "  export KUBECONFIG=$PWD/infra/kubeconfig.yaml AIRFLOW_PASSWORD=<AIRFLOW_ADMIN_PASSWORD> GROQ_API_KEY=<key>"
echo "  python -m agent.main"
