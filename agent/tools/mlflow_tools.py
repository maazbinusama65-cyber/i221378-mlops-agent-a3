"""MLflow experiment-intelligence tools (MlflowClient only - the web UI is never scraped)."""

from __future__ import annotations

import os
import statistics
from datetime import UTC, datetime
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from requests.exceptions import RequestException

from agent.config import Settings
from agent.schemas import (
    MlflowBestArgs,
    MlflowCompareExperimentsArgs,
    MlflowCompareRunsArgs,
    MlflowListRunsArgs,
    MlflowNoArgs,
    MlflowRegistryArgs,
    MlflowRunArgs,
)
from agent.tools.base import Tool, ToolError, ToolUnavailable

KEY_METRICS = ("f1", "precision", "recall", "roc_auc", "train_f1")
PROVENANCE_TAGS = ("dvc.dataset_version", "dvc.md5", "dvc.dirty", "git.commit", "quality_gate", "trigger", "airflow.dag_run_id")
OVERFIT_GAP = 0.10
SMALL_TEST_SET = 300
TIE_MARGIN = 0.02


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds") if ms else None


def _row(run: Any) -> dict[str, Any]:
    m, t = run.data.metrics, run.data.tags
    return {
        "run_id": run.info.run_id,
        "name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "started": _iso(run.info.start_time),
        "metrics": {k: round(m[k], 4) for k in KEY_METRICS if k in m},
        "dataset_version": t.get("dvc.dataset_version"),
        "algorithm": run.data.params.get("algorithm"),
        "quality_gate": t.get("quality_gate"),
    }


class MlflowTools:
    def __init__(self, settings: Settings) -> None:
        # Fail fast instead of MLflow's default 7 retries with exponential backoff.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
        self.client = MlflowClient(tracking_uri=settings.mlflow_uri, registry_uri=settings.mlflow_uri)
        self.uri = settings.mlflow_uri

    # -- plumbing --------------------------------------------------------------------------
    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except RequestException as exc:
            raise ToolUnavailable(f"MLflow at {self.uri} unreachable: {type(exc).__name__}") from exc
        except MlflowException as exc:
            text = str(exc)
            if any(k in text for k in ("Max retries", "Connection", "timed out", "Failed to establish")):
                raise ToolUnavailable(f"MLflow at {self.uri} unreachable: {text[:160]}") from exc
            raise ToolError(f"MLflow error: {text[:300]}") from exc

    def _experiment_ids(self, name: str | None) -> list[str]:
        if name is None:
            return [e.experiment_id for e in self._call(self.client.search_experiments)]
        exp = self._call(self.client.get_experiment_by_name, name)
        if exp is None:
            raise ToolError(f"experiment {name!r} not found")
        return [exp.experiment_id]

    def _recent_runs(self, experiment: str | None, n: int) -> list[Any]:
        ids = self._experiment_ids(experiment)
        runs = self._call(self.client.search_runs, ids, filter_string="attributes.status = 'FINISHED'",
                          order_by=["attributes.start_time DESC"], max_results=n)
        return list(runs)

    # -- tools -----------------------------------------------------------------------------
    def list_experiments(self, _: MlflowNoArgs) -> dict[str, Any]:
        exps = self._call(self.client.search_experiments)
        out = []
        for e in exps:
            runs = self._call(self.client.search_runs, [e.experiment_id], max_results=200)
            out.append({"experiment_id": e.experiment_id, "name": e.name, "runs": len(runs),
                        "last_updated": _iso(e.last_update_time)})
        return {"tracking_uri": self.uri, "experiments": out}

    def list_runs(self, args: MlflowListRunsArgs) -> dict[str, Any]:
        runs = self._call(self.client.search_runs, self._experiment_ids(args.experiment),
                          order_by=["attributes.start_time DESC"], max_results=args.limit)
        return {"count": len(runs), "runs": [_row(r) for r in runs]}

    def get_run(self, args: MlflowRunArgs) -> dict[str, Any]:
        run = self._call(self.client.get_run, args.run_id)
        arts = [a.path for a in self._call(self.client.list_artifacts, args.run_id)]
        versions = self._call(self.client.search_model_versions, f"run_id='{args.run_id}'")
        data = _row(run)
        data.update(
            params=dict(run.data.params),
            metrics={k: round(v, 4) for k, v in run.data.metrics.items()},
            provenance={k: run.data.tags[k] for k in PROVENANCE_TAGS if k in run.data.tags},
            artifacts=arts,
            registered_versions=[{"name": v.name, "version": v.version} for v in versions],
        )
        return data

    def best_run(self, args: MlflowBestArgs) -> dict[str, Any]:
        runs = [r for r in self._recent_runs(args.experiment, args.recent) if args.metric in r.data.metrics]
        if not runs:
            raise ToolError(f"no finished runs with metric {args.metric!r}")
        ranked = sorted(runs, key=lambda r: r.data.metrics[args.metric], reverse=True)
        best, others = ranked[0], ranked[1:]
        gap = best.data.metrics.get("train_f1", best.data.metrics[args.metric]) - best.data.metrics.get("f1", 0)
        t = best.data.tags
        caveats = []
        if gap > OVERFIT_GAP:
            caveats.append(f"train/test F1 gap {gap:.2f} > {OVERFIT_GAP}: possible overfitting")
        if t.get("quality_gate") != "passed":
            caveats.append(f"quality gate status is {t.get('quality_gate')!r}")
        if t.get("dvc.dirty") == "true":
            caveats.append("trained on a dataset that differed from its recorded DVC version")
        if t.get("dvc.dataset_version") in (None, "untracked"):
            caveats.append("no DVC dataset version recorded")
        if best.data.metrics.get("test_rows", SMALL_TEST_SET) < SMALL_TEST_SET:
            caveats.append("small evaluation set")
        if others and best.data.metrics[args.metric] - others[0].data.metrics[args.metric] < TIE_MARGIN:
            caveats.append(f"margin over runner-up < {TIE_MARGIN}: difference may be within noise")
        registered = self._call(self.client.search_model_versions, f"run_id='{best.info.run_id}'")
        if not registered:
            caveats.append("model is not registered in the model registry")
        return {
            "metric": args.metric,
            "considered_runs": len(runs),
            "best": _row(best) | {"git_commit": t.get("git.commit"), "dvc_md5": t.get("dvc.md5")},
            "ranking": [{"run_id": r.info.run_id, "name": r.info.run_name, args.metric: round(r.data.metrics[args.metric], 4),
                         "dataset_version": r.data.tags.get("dvc.dataset_version")} for r in ranked[:5]],
            "caveats": caveats,
            "production_ready": "not asserted: the top metric alone does not establish production readiness "
            "(needs the caveats resolved plus deployment/serving checks)",
        }

    def compare_runs(self, args: MlflowCompareRunsArgs) -> dict[str, Any]:
        if args.run_ids:
            runs = [self._call(self.client.get_run, rid) for rid in args.run_ids]
        elif args.last_n:
            runs = self._recent_runs(args.experiment, args.last_n)
        else:
            raise ToolError("provide run_ids or last_n")
        if len(runs) < 2:
            raise ToolError("need at least two runs to compare")
        rows = [_row(r) | {"params": dict(r.data.params)} for r in runs]
        winners = {}
        for k in KEY_METRICS:
            scored = [r for r in runs if k in r.data.metrics]
            if scored:
                top = max(scored, key=lambda r: r.data.metrics[k])
                winners[k] = {"run_id": top.info.run_id, "name": top.info.run_name, "value": round(top.data.metrics[k], 4)}
        return {"runs": rows, "winner_per_metric": winners,
                "note": "metrics come from different datasets if dataset_version differs; compare with care"}

    def compare_experiments(self, args: MlflowCompareExperimentsArgs) -> dict[str, Any]:
        out = []
        for name in args.experiments:
            runs = [r for r in self._recent_runs(name, 100) if args.metric in r.data.metrics]
            if not runs:
                out.append({"experiment": name, "runs": 0})
                continue
            vals = [r.data.metrics[args.metric] for r in runs]
            best = max(runs, key=lambda r: r.data.metrics[args.metric])
            out.append({"experiment": name, "runs": len(runs), "best": round(max(vals), 4), "mean": round(statistics.fmean(vals), 4),
                        "best_run": {"run_id": best.info.run_id, "name": best.info.run_name},
                        "dataset_versions": sorted({r.data.tags.get("dvc.dataset_version", "?") for r in runs})})
        return {"metric": args.metric, "experiments": out}

    def registry(self, args: MlflowRegistryArgs) -> dict[str, Any]:
        try:
            model = self._call(self.client.get_registered_model, args.name)
        except ToolError as exc:
            raise ToolError(f"registered model {args.name!r} not found ({exc})") from exc
        aliases = dict(getattr(model, "aliases", {}) or {})
        versions = []
        for v in sorted(self._call(self.client.search_model_versions, f"name='{args.name}'"), key=lambda v: int(v.version)):
            run = self._call(self.client.get_run, v.run_id) if v.run_id else None
            versions.append({
                "version": v.version,
                "aliases": [a for a, ver in aliases.items() if str(ver) == str(v.version)],
                "run_id": v.run_id,
                "run_name": run.info.run_name if run else None,
                "f1": round(run.data.metrics.get("f1", float("nan")), 4) if run else None,
                "dataset_version": run.data.tags.get("dvc.dataset_version") if run else None,
            })
        return {"name": args.name, "aliases": aliases, "versions": versions}


def build_tools(settings: Settings) -> list[Tool]:
    m = MlflowTools(settings)
    return [
        Tool("mlflow_list_experiments", "mlflow", "List MLflow experiments with run counts.", MlflowNoArgs, m.list_experiments),
        Tool("mlflow_list_runs", "mlflow", "List recent runs (newest first) with key metrics and dataset version.", MlflowListRunsArgs, m.list_runs),
        Tool("mlflow_get_run", "mlflow", "Full params, metrics, DVC/Git provenance tags, artifacts and registry versions of one run.", MlflowRunArgs, m.get_run),
        Tool("mlflow_best_run", "mlflow", "Best recent run by a metric, with ranking and production-readiness caveats.", MlflowBestArgs, m.best_run),
        Tool("mlflow_compare_runs", "mlflow", "Compare runs side by side with winner per metric and params. USE THIS for 'compare the last N runs' (last_n) or specific run_ids.", MlflowCompareRunsArgs, m.compare_runs),
        Tool("mlflow_compare_experiments", "mlflow", "Compare two or more experiments on a metric.", MlflowCompareExperimentsArgs, m.compare_experiments),
        Tool("mlflow_model_registry", "mlflow", "Registered model versions, aliases (production/candidate) and source runs.", MlflowRegistryArgs, m.registry),
    ]
