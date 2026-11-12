"""Apache Airflow 3 workflow tools (REST API v2 with JWT bearer auth)."""

from __future__ import annotations

import re
import threading
from typing import Any

import httpx
from pydantic import BaseModel

from agent.config import Settings
from agent.logging_utils import register_secret
from agent.schemas import AirflowLogsArgs, AirflowNoArgs, AirflowRunArgs, AirflowRunsArgs, AirflowTriggerArgs
from agent.tools.base import Tool, ToolError, ToolRejected, ToolUnavailable

FAILED_STATES = {"failed", "upstream_failed"}
# Parameters an operator may override when triggering: numeric ranges, or a tuple of allowed values.
ALLOWED_CONF: dict[str, tuple] = {
    "min_f1": (0.0, 1.0), "min_recall": (0.0, 1.0), "n_estimators": (10, 500), "max_depth": (1, 30),
    "algorithm": (("random_forest", "logistic_regression"), None),
}
_DATA_PATH_RE = re.compile(r"data/[A-Za-z0-9_./\-]+\.csv")


class AirflowClient:
    """Minimal Airflow 3 client: fetches a JWT from /auth/token and refreshes it on 401."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.s = settings
        self._http = httpx.Client(base_url=settings.airflow_url, timeout=8.0, transport=transport)
        self._token: str | None = None
        self._lock = threading.Lock()
        register_secret(settings.airflow_password)

    def _login(self) -> str:
        if not self.s.airflow_password:
            raise ToolUnavailable("AIRFLOW_PASSWORD is not configured")
        resp = self._http.post("/auth/token", json={"username": self.s.airflow_user, "password": self.s.airflow_password})
        if resp.status_code not in (200, 201):
            raise ToolError(f"Airflow authentication failed (HTTP {resp.status_code})")
        token = resp.json()["access_token"]
        register_secret(token)
        return token

    def request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            for attempt in (1, 2):
                with self._lock:
                    if self._token is None:
                        self._token = self._login()
                    token = self._token
                resp = self._http.request(method, f"/api/v2{path}", headers={"Authorization": f"Bearer {token}"}, **kw)
                if resp.status_code == 401 and attempt == 1:
                    with self._lock:
                        if self._token == token:
                            self._token = None
                    continue
                break
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ToolUnavailable(f"Airflow at {self.s.airflow_url} unreachable: {type(exc).__name__}") from exc
        if resp.status_code == 404:
            raise ToolError(f"not found: {path}")
        if resp.status_code >= 400:
            raise ToolError(f"Airflow API {resp.status_code}: {resp.text[:200]}")
        return resp.json()


def _run_row(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date")} | {
        "triggered_by": r.get("triggered_by"), "conf": r.get("conf")}


def _ti_row(t: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": t["task_id"], "state": t.get("state"), "try_number": t.get("try_number"),
            "max_tries": t.get("max_tries"), "duration_s": round(t["duration"], 1) if t.get("duration") else None}


class AirflowTools:
    def __init__(self, settings: Settings, client: AirflowClient | None = None) -> None:
        self.s = settings
        self.api = client or AirflowClient(settings)

    def _approved(self, dag_id: str) -> None:
        if dag_id not in self.s.approved_dags:
            raise ToolRejected(f"DAG {dag_id!r} is not on the approved list {list(self.s.approved_dags)}")

    def _resolve_run(self, dag_id: str, run_id: str | None) -> dict[str, Any]:
        if run_id:
            return self.api.request("GET", f"/dags/{dag_id}/dagRuns/{run_id}")
        runs = self.api.request("GET", f"/dags/{dag_id}/dagRuns", params={"order_by": "-start_date", "limit": 1})["dag_runs"]
        if not runs:
            raise ToolError(f"no runs recorded for DAG {dag_id}")
        return runs[0]

    # -- tools -----------------------------------------------------------------------------
    def list_dags(self, _: AirflowNoArgs) -> dict[str, Any]:
        dags = self.api.request("GET", "/dags")["dags"]
        return {"dags": [{"dag_id": d["dag_id"], "paused": d["is_paused"], "approved_to_trigger": d["dag_id"] in self.s.approved_dags,
                          "tags": [t["name"] for t in d.get("tags", [])]} for d in dags]}

    def dag_runs(self, args: AirflowRunsArgs) -> dict[str, Any]:
        runs = self.api.request("GET", f"/dags/{args.dag_id}/dagRuns", params={"order_by": "-start_date", "limit": args.limit})["dag_runs"]
        return {"dag_id": args.dag_id, "runs": [_run_row(r) for r in runs]}

    def task_instances(self, args: AirflowRunArgs) -> dict[str, Any]:
        run = self._resolve_run(args.dag_id, args.run_id)
        tis = self.api.request("GET", f"/dags/{args.dag_id}/dagRuns/{run['dag_run_id']}/taskInstances")["task_instances"]
        return {"dag_id": args.dag_id, "run": _run_row(run), "tasks": [_ti_row(t) for t in tis]}

    def _raw_logs(self, dag_id: str, run_id: str, task_id: str, try_number: int) -> list[str]:
        payload = self.api.request("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}")
        content = payload.get("content", []) if isinstance(payload, dict) else []
        out = []
        for item in content:
            if isinstance(item, dict):
                ts = (item.get("timestamp") or "")[11:19]
                out.append(f"{ts} {item.get('level', '')} {item.get('event', '')}".strip())
                for exc in item.get("error_detail") or []:
                    out.append(f"EXCEPTION {exc.get('exc_type')}: {str(exc.get('exc_value'))[:300]}")
            else:
                out.append(str(item))
        return out

    def failed_tasks(self, args: AirflowRunArgs) -> dict[str, Any]:
        if args.run_id:
            run = self._resolve_run(args.dag_id, args.run_id)
        else:
            runs = self.api.request("GET", f"/dags/{args.dag_id}/dagRuns", params={"order_by": "-start_date", "limit": 10})["dag_runs"]
            failed = [r for r in runs if r["state"] == "failed"]
            if not failed:
                return {"dag_id": args.dag_id, "failed_run": None, "message": "no failed runs among the 10 most recent",
                        "latest_state": runs[0]["state"] if runs else None}
            run = failed[0]
        tis = self.api.request("GET", f"/dags/{args.dag_id}/dagRuns/{run['dag_run_id']}/taskInstances")["task_instances"]
        failed_tis = [t for t in tis if t.get("state") in FAILED_STATES]
        details = []
        for t in failed_tis:
            entry = _ti_row(t)
            if t["state"] == "failed":
                try:
                    entry["log_tail"] = self._raw_logs(args.dag_id, run["dag_run_id"], t["task_id"], max(int(t.get("try_number") or 1), 1))[-8:]
                except ToolError as exc:
                    entry["log_tail"] = [f"logs unavailable: {exc}"]
            details.append(entry)
        return {"dag_id": args.dag_id, "failed_run": _run_row(run), "failed_tasks": details,
                "root_cause_task": next((d["task_id"] for d in details if d["state"] == "failed"), None),
                "all_tasks": [f"{t['task_id']}={t.get('state')}" for t in tis]}

    def task_logs(self, args: AirflowLogsArgs) -> dict[str, Any]:
        run = self._resolve_run(args.dag_id, args.run_id)
        try_number = args.try_number or 1
        lines = self._raw_logs(args.dag_id, run["dag_run_id"], args.task_id, try_number)
        return {"dag_id": args.dag_id, "run_id": run["dag_run_id"], "task_id": args.task_id, "try_number": try_number,
                "log_tail": lines[-args.tail_lines:], "total_lines": len(lines)}

    def trigger(self, args: AirflowTriggerArgs) -> dict[str, Any]:
        self._approved(args.dag_id)
        run = self.api.request("POST", f"/dags/{args.dag_id}/dagRuns", json={"logical_date": None, "conf": args.conf or {}})
        return {"triggered": True, "dag_id": args.dag_id, **_run_row(run)}

    # -- safety hooks ----------------------------------------------------------------------
    def guard_trigger(self, args: BaseModel) -> None:
        self._approved(args.dag_id)  # type: ignore[attr-defined]
        for key, value in (args.conf or {}).items():  # type: ignore[attr-defined]
            if key == "data_path":
                if not (isinstance(value, str) and _DATA_PATH_RE.fullmatch(value)) or ".." in value:
                    raise ToolRejected(f"DAG conf data_path={value!r} must be a relative data/**.csv path without '..'")
                continue
            if key not in ALLOWED_CONF:
                raise ToolRejected(f"DAG conf key {key!r} is not allowed; permitted: {sorted(ALLOWED_CONF)}")
            low, high = ALLOWED_CONF[key]
            if isinstance(low, (int, float)) and not (isinstance(value, (int, float)) and low <= value <= high):
                raise ToolRejected(f"DAG conf {key}={value!r} outside the permitted range [{low}, {high}]")
            if isinstance(low, tuple) and value not in low:
                raise ToolRejected(f"DAG conf {key}={value!r} not in {list(low)}")

    @staticmethod
    def describe_trigger(args: BaseModel) -> str:
        return f"Trigger Airflow DAG '{args.dag_id}' with conf={args.conf or {}}"  # type: ignore[attr-defined]


def build_tools(settings: Settings, client: AirflowClient | None = None) -> list[Tool]:
    a = AirflowTools(settings, client)
    return [
        Tool("airflow_list_dags", "airflow", "List available Airflow DAGs and which are approved to trigger.", AirflowNoArgs, a.list_dags),
        Tool("airflow_dag_runs", "airflow", "Recent DAG runs with state (success/failed/running).", AirflowRunsArgs, a.dag_runs),
        Tool("airflow_task_instances", "airflow", "Per-task states, tries and durations for a DAG run (latest if run_id omitted).", AirflowRunArgs, a.task_instances),
        Tool("airflow_failed_tasks", "airflow", "Identify failed tasks of the latest failed run, with log excerpts for diagnosis.", AirflowRunArgs, a.failed_tasks),
        Tool("airflow_task_logs", "airflow", "Fetch the log tail of one task instance.", AirflowLogsArgs, a.task_logs),
        Tool("airflow_trigger_dag", "airflow", "Trigger an APPROVED training DAG. Requires user confirmation; never call unless the user asked to run training.",
             AirflowTriggerArgs, a.trigger, access="write", guard=a.guard_trigger, describe=a.describe_trigger),
    ]
