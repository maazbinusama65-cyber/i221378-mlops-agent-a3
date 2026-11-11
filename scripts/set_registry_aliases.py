"""Set registry aliases after seeding: v1-trained model is what is deployed ('production'),
the newest dataset's logistic regression is the 'candidate'."""

from __future__ import annotations

import os

from mlflow.tracking import MlflowClient

NAME = "churn-classifier"


def main() -> None:
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    versions = sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version))
    for v in versions:
        run = client.get_run(v.run_id)
        print(v.version, run.info.run_name, run.data.tags.get("dvc.dataset_version"), round(run.data.metrics["f1"], 4))
    client.set_registered_model_alias(NAME, "production", versions[0].version)
    client.set_registered_model_alias(NAME, "candidate", versions[-1].version)
    print(f"production -> v{versions[0].version}, candidate -> v{versions[-1].version}")


if __name__ == "__main__":
    main()
