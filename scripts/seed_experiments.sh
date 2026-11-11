#!/usr/bin/env bash
# Seed MLflow with real training runs on both DVC dataset versions.
# Requires: venv active, MLflow at $MLFLOW_TRACKING_URI (default http://127.0.0.1:5000), DVC remote populated.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUTF8=1

use_dataset() {  # $1 = git tag (data-v1 | data-v2); restores the exact .dvc file and data via DVC
  git checkout "$1" -- data/churn.csv.dvc
  dvc checkout data/churn.csv.dvc --force
}

train() { python -m training.train "$@" 2>&1 | grep -E "^run_id="; }

echo "== dataset v1 =="
use_dataset data-v1
train --experiment churn-baselines --run-name v1-logreg --algorithm logistic_regression --C 1.0
train --experiment churn-baselines --run-name v1-rf-d4 --n-estimators 50 --max-depth 4
train --experiment churn-baselines --run-name v1-rf-d8 --n-estimators 100 --max-depth 8
train --experiment churn-training --run-name v1-rf-d6-registered --n-estimators 150 --max-depth 6 --register --export models/model.joblib

echo "== dataset v2 =="
use_dataset data-v2
train --experiment churn-baselines --run-name v2-logreg --algorithm logistic_regression --C 0.5
train --experiment churn-training --run-name v2-rf-d4 --n-estimators 100 --max-depth 4
train --experiment churn-training --run-name v2-rf-d6 --n-estimators 200 --max-depth 6
train --experiment churn-training --run-name v2-rf-d10-balanced --n-estimators 200 --max-depth 10 --class-weight balanced
train --experiment churn-training --run-name v2-rf-overfit --n-estimators 300 --register
train --experiment churn-training --run-name v2-logreg-balanced --algorithm logistic_regression --C 2.0 --class-weight balanced --register

git checkout HEAD -- data/churn.csv.dvc
dvc checkout data/churn.csv.dvc --force
echo "done; working tree restored to HEAD dataset"
