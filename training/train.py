"""CLI training entrypoint: ``python -m training.train --algorithm random_forest --n-estimators 200``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from training import core

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and track a churn model")
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--algorithm", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    parser.add_argument("--experiment", default=core.DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--register", action="store_true", help="register the model in the MLflow registry")
    parser.add_argument("--export", default=None, help="also write the model to this joblib path (for serving)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    data = REPO_ROOT / args.data
    core.validate_dataset(data)
    splits = core.preprocess(data, REPO_ROOT / "data" / "processed")
    params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "C": args.C, "class_weight": args.class_weight}
    params = {k: v for k, v in params.items() if v is not None}
    model, metrics = core.train_and_evaluate(splits, args.algorithm, params)
    violations = core.check_quality_gate(metrics, core.QualityGate())
    run_id = core.log_run(
        model,
        metrics,
        algorithm=args.algorithm,
        params=params,
        data_path=data,
        repo_root=REPO_ROOT,
        experiment=args.experiment,
        run_name=args.run_name,
        register=args.register,
        gate_result=violations,
        extra_tags={"trigger": "cli"},
    )
    if args.export:
        out = REPO_ROOT / args.export
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
    print(f"run_id={run_id} f1={metrics['f1']} recall={metrics['recall']} gate={'PASS' if not violations else 'FAIL'}")


if __name__ == "__main__":
    main()
