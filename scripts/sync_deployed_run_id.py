"""Point the ConfigMap (and the model's DVC metadata) at the MLflow run that produced the serving model.

On a fresh clone the seeded MLflow run ids differ from the ones committed in the repo, so this rewrites
MODEL_RUN_ID / mlflow_run_id with the real id of the run named ``v1-rf-d6-registered``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "v1-rf-d6-registered"


def main() -> None:
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    experiment = client.get_experiment_by_name("churn-training")
    runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{RUN_NAME}'", max_results=1)
    if not runs:
        raise SystemExit(f"run {RUN_NAME!r} not found - run scripts/seed_experiments.sh first")
    run_id = runs[0].info.run_id
    for rel, pattern in (("kubernetes/configmap.yaml", r'(MODEL_RUN_ID: ")[0-9a-f]{32}(")'),
                         ("models/model.joblib.dvc", r"(mlflow_run_id: )[0-9a-f]{32}()")):
        path = ROOT / rel
        path.write_text(re.sub(pattern, rf"\g<1>{run_id}\g<2>", path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"deployed model run id -> {run_id}")


if __name__ == "__main__":
    main()
