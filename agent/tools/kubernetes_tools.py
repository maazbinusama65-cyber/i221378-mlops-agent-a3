"""Kubernetes operations tools.

Safety model
  * Every tool is bound to an approved-namespace allowlist (default: ``mlops-agent``) - reads included.
  * Cluster-changing tools (restart / scale) are ``access="write"``: the registry never runs them from an LLM
    request; it returns ``confirmation_required`` and only a human ``/confirm <id>`` executes the action.
  * Scale is bounded (0..``k8s_max_replicas``); pod log output is redacted before it reaches the model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from pydantic import BaseModel
from urllib3.exceptions import HTTPError as Urllib3Error

from agent.config import Settings
from agent.logging_utils import redact_text
from agent.schemas import K8sDeploymentArgs, K8sEventsArgs, K8sLogsArgs, K8sNsArgs, K8sPodArgs, K8sScaleArgs
from agent.tools.base import Tool, ToolError, ToolRejected, ToolUnavailable
from kubernetes import client, config

REQUEST_TIMEOUT = (3.0, 10.0)  # connect, read


RECENT_EVENT_WINDOW_S = 900  # diagnosis only treats warning events from the last 15 minutes as current


def _age_s(ts: datetime | None) -> int | None:
    return None if ts is None else int((datetime.now(UTC) - ts).total_seconds())


def _age(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    secs = int((datetime.now(UTC) - ts).total_seconds())
    return f"{secs // 3600}h{secs % 3600 // 60}m" if secs >= 3600 else f"{secs // 60}m{secs % 60}s"


class K8sTools:
    def __init__(self, settings: Settings, core: client.CoreV1Api | None = None, apps: client.AppsV1Api | None = None) -> None:
        self.s = settings
        self._core, self._apps = core, apps

    # -- plumbing --------------------------------------------------------------------------
    @property
    def core(self) -> client.CoreV1Api:
        if self._core is None:
            self._load()
        return self._core  # type: ignore[return-value]

    @property
    def apps(self) -> client.AppsV1Api:
        if self._apps is None:
            self._load()
        return self._apps  # type: ignore[return-value]

    def _load(self) -> None:
        try:
            config.load_kube_config(config_file=self.s.kubeconfig)
        except (ConfigException, OSError) as exc:
            raise ToolUnavailable(f"no usable kubeconfig: {exc}") from exc
        self._core, self._apps = client.CoreV1Api(), client.AppsV1Api()

    def _ns(self, namespace: str) -> str:
        if namespace not in self.s.k8s_namespaces:
            raise ToolRejected(f"namespace {namespace!r} is not approved; allowed: {list(self.s.k8s_namespaces)}")
        return namespace

    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, _request_timeout=REQUEST_TIMEOUT, **kw)
        except ApiException as exc:
            if exc.status == 404:
                raise ToolError(f"not found: {exc.reason}") from exc
            if exc.status in (401, 403):
                raise ToolRejected(f"Kubernetes denied access (HTTP {exc.status}): {exc.reason}") from exc
            raise ToolError(f"Kubernetes API error {exc.status}: {exc.reason}") from exc
        except (Urllib3Error, OSError) as exc:
            raise ToolUnavailable(f"Kubernetes API unreachable: {type(exc).__name__}") from exc

    # -- read tools ------------------------------------------------------------------------
    def list_deployments(self, args: K8sNsArgs) -> dict[str, Any]:
        items = self._call(self.apps.list_namespaced_deployment, self._ns(args.namespace)).items
        return {"namespace": args.namespace, "deployments": [{
            "name": d.metadata.name, "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0,
            "available": d.status.available_replicas or 0, "updated": d.status.updated_replicas or 0,
            "image": d.spec.template.spec.containers[0].image,
            "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        } for d in items]}

    def list_pods(self, args: K8sNsArgs) -> dict[str, Any]:
        pods = self._call(self.core.list_namespaced_pod, self._ns(args.namespace)).items
        return {"namespace": args.namespace, "pods": [self._pod_row(p) for p in pods]}

    @staticmethod
    def _waiting_reason(p: Any) -> str | None:
        for cs in p.status.container_statuses or []:
            if cs.state and cs.state.waiting:
                return cs.state.waiting.reason
        return None

    def _pod_row(self, p: Any) -> dict[str, Any]:
        statuses = p.status.container_statuses or []
        return {
            "name": p.metadata.name, "phase": p.status.phase,
            "ready": f"{sum(1 for c in statuses if c.ready)}/{len(statuses)}",
            "restarts": sum(c.restart_count for c in statuses), "age": _age(p.metadata.creation_timestamp),
            "waiting_reason": self._waiting_reason(p),
        }

    def pod_status(self, args: K8sPodArgs) -> dict[str, Any]:
        p = self._call(self.core.read_namespaced_pod, args.pod, self._ns(args.namespace))
        containers = []
        for cs in p.status.container_statuses or []:
            state = "running" if cs.state.running else ("waiting" if cs.state.waiting else "terminated")
            entry: dict[str, Any] = {"name": cs.name, "ready": cs.ready, "restart_count": cs.restart_count, "state": state, "image": cs.image}
            if cs.state.waiting:
                entry["waiting_reason"], entry["waiting_message"] = cs.state.waiting.reason, (cs.state.waiting.message or "")[:200]
            if cs.last_state and cs.last_state.terminated:
                t = cs.last_state.terminated
                entry["last_terminated"] = {"reason": t.reason, "exit_code": t.exit_code, "finished_at": str(t.finished_at)}
            containers.append(entry)
        spec = p.spec.containers[0]
        probes = {"readiness": getattr(spec.readiness_probe.http_get, "path", None) if spec.readiness_probe else None,
                  "liveness": getattr(spec.liveness_probe.http_get, "path", None) if spec.liveness_probe else None}
        res = spec.resources
        return {
            "pod": p.metadata.name, "phase": p.status.phase, "node": p.spec.node_name, "age": _age(p.metadata.creation_timestamp),
            "conditions": [{"type": c.type, "status": c.status, "reason": c.reason, "message": (c.message or "")[:160]}
                           for c in p.status.conditions or [] if c.status != "True"] or "all conditions True",
            "containers": containers, "probes": probes,
            "resources": {"requests": (res.requests or {}), "limits": (res.limits or {})},
        }

    def pod_logs(self, args: K8sLogsArgs) -> dict[str, Any]:
        text = self._call(self.core.read_namespaced_pod_log, args.pod, self._ns(args.namespace),
                          tail_lines=args.tail_lines, previous=args.previous)
        lines = redact_text(text).splitlines()
        return {"pod": args.pod, "previous": args.previous, "lines": lines[-args.tail_lines:], "line_count": len(lines)}

    def services(self, args: K8sNsArgs) -> dict[str, Any]:
        ns = self._ns(args.namespace)
        out = []
        for svc in self._call(self.core.list_namespaced_service, ns).items:
            try:
                ep = self._call(self.core.read_namespaced_endpoints, svc.metadata.name, ns)
                subsets = ep.subsets or []
                ready = sum(len(s.addresses or []) for s in subsets)
                not_ready = sum(len(s.not_ready_addresses or []) for s in subsets)
            except ToolError:
                ready = not_ready = 0
            out.append({"name": svc.metadata.name, "type": svc.spec.type, "cluster_ip": svc.spec.cluster_ip,
                        "ports": [f"{p.port}->{p.target_port}" for p in svc.spec.ports or []], "selector": svc.spec.selector,
                        "ready_endpoints": ready, "not_ready_endpoints": not_ready})
        return {"namespace": ns, "services": out}

    def events(self, args: K8sEventsArgs) -> dict[str, Any]:
        items = self._call(self.core.list_namespaced_event, self._ns(args.namespace)).items
        if args.warnings_only:
            items = [e for e in items if e.type == "Warning"]
        items.sort(key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp, reverse=True)
        return {"namespace": args.namespace, "events": [{
            "type": e.type, "reason": e.reason, "object": f"{e.involved_object.kind}/{e.involved_object.name}",
            "message": (e.message or "")[:220], "count": e.count, "age": _age(e.last_timestamp or e.metadata.creation_timestamp),
            "age_s": _age_s(e.last_timestamp or e.metadata.creation_timestamp),
        } for e in items[:15]]}

    def app_version(self, args: K8sDeploymentArgs) -> dict[str, Any]:
        ns = self._ns(args.namespace)
        dep = self._call(self.apps.read_namespaced_deployment, args.deployment, ns)
        container = dep.spec.template.spec.containers[0]
        cm_values: dict[str, str] = {}
        for src in container.env_from or []:
            if src.config_map_ref:
                cm = self._call(self.core.read_namespaced_config_map, src.config_map_ref.name, ns)
                cm_values.update(cm.data or {})
        keys = ("APP_VERSION", "MODEL_RUN_ID", "DATASET_VERSION", "MODEL_PATH")
        return {"deployment": args.deployment, "image": container.image, "version_label": (dep.metadata.labels or {}).get("app.kubernetes.io/version"),
                "generation": dep.metadata.generation, "observed_generation": dep.status.observed_generation,
                "config": {k: cm_values.get(k) for k in keys}, "replicas": {"desired": dep.spec.replicas, "ready": dep.status.ready_replicas or 0}}

    def diagnose_deployment(self, args: K8sDeploymentArgs) -> dict[str, Any]:
        ns = self._ns(args.namespace)
        dep = self._call(self.apps.read_namespaced_deployment, args.deployment, ns)
        selector = ",".join(f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items())
        pods = self._call(self.core.list_namespaced_pod, ns, label_selector=selector).items
        desired, ready = dep.spec.replicas or 0, dep.status.ready_replicas or 0
        unhealthy, hints = [], []
        for p in pods:
            row = self._pod_row(p)
            bad = p.status.phase != "Running" or row["ready"].split("/")[0] != row["ready"].split("/")[1] or row["restarts"] > 0
            if not bad:
                continue
            entry = dict(row)
            try:
                entry["log_tail"] = self.pod_logs(K8sLogsArgs(pod=p.metadata.name, namespace=ns, tail_lines=6, previous=row["restarts"] > 0 and not row["ready"].startswith("1/"))).get("lines", [])
            except (ToolError, ToolRejected):
                entry["log_tail"] = []
            unhealthy.append(entry)
            hints.extend(self._hints(p, entry))
        all_warnings = self.events(K8sEventsArgs(namespace=ns, warnings_only=True))["events"]
        warnings = [e for e in all_warnings if (e["age_s"] or 0) <= RECENT_EVENT_WINDOW_S][:8]
        older_warnings = len(all_warnings) - len(warnings)
        return {
            "deployment": args.deployment, "desired": desired, "ready": ready, "healthy": desired > 0 and ready == desired,
            "pods_total": len(pods), "unhealthy_pods": unhealthy[:4], "warning_events": warnings,
            "older_warning_events_ignored": older_warnings,
            "likely_causes": sorted(set(hints)) or ["no fault detected in pod/probe state"],
        }

    @staticmethod
    def _hints(p: Any, entry: dict[str, Any]) -> list[str]:
        hints = []
        waiting = entry.get("waiting_reason")
        if waiting in ("ImagePullBackOff", "ErrImagePull", "ErrImageNeverPull"):
            hints.append(f"image cannot be pulled ({waiting}): wrong image name/tag or registry access")
        if waiting == "CrashLoopBackOff":
            hints.append("container is crash-looping: inspect previous logs for the exit cause")
        for cs in p.status.container_statuses or []:
            if cs.last_state and cs.last_state.terminated and cs.last_state.terminated.reason == "OOMKilled":
                hints.append("container was OOMKilled: memory limit too low")
        if p.status.phase == "Running" and not entry["ready"].startswith(entry["ready"].split("/")[1] + "/") and entry["restarts"] == 0:
            hints.append("pod Running but NOT Ready: readiness probe is failing (app up, dependency/config broken)")
        if p.status.phase == "Pending":
            hints.append("pod Pending: unschedulable (resources/node) or waiting for image")
        text = " ".join(entry.get("log_tail", [])).lower()
        if "model load failed" in text or "no such file" in text:
            hints.append("application log shows the model file could not be loaded (check MODEL_PATH in the ConfigMap / image contents)")
        return hints

    # -- write tools (confirmation-gated, namespace-bound) -----------------------------------
    def restart(self, args: K8sDeploymentArgs) -> dict[str, Any]:
        ns = self._ns(args.namespace)
        body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": datetime.now(UTC).isoformat()}}}}}
        self._call(self.apps.patch_namespaced_deployment, args.deployment, ns, body)
        return {"restarted": args.deployment, "namespace": ns, "method": "rolling restart via pod-template annotation"}

    def scale(self, args: K8sScaleArgs) -> dict[str, Any]:
        ns = self._ns(args.namespace)
        self._call(self.apps.patch_namespaced_deployment_scale, args.deployment, ns, {"spec": {"replicas": args.replicas}})
        return {"scaled": args.deployment, "namespace": ns, "replicas": args.replicas}

    # -- safety hooks ----------------------------------------------------------------------
    def guard_restart(self, args: BaseModel) -> None:
        self._ns(args.namespace)  # type: ignore[attr-defined]

    def guard_scale(self, args: BaseModel) -> None:
        self._ns(args.namespace)  # type: ignore[attr-defined]
        if args.replicas > self.s.k8s_max_replicas:  # type: ignore[attr-defined]
            raise ToolRejected(f"replicas {args.replicas} exceeds the allowed maximum {self.s.k8s_max_replicas}")  # type: ignore[attr-defined]

    @staticmethod
    def describe_restart(args: BaseModel) -> str:
        return f"Rolling-restart deployment '{args.deployment}' in namespace '{args.namespace}'"  # type: ignore[attr-defined]

    @staticmethod
    def describe_scale(args: BaseModel) -> str:
        return f"Scale deployment '{args.deployment}' in namespace '{args.namespace}' to {args.replicas} replicas"  # type: ignore[attr-defined]


def build_tools(settings: Settings, core: client.CoreV1Api | None = None, apps: client.AppsV1Api | None = None) -> list[Tool]:
    k = K8sTools(settings, core, apps)
    return [
        Tool("k8s_list_deployments", "kubernetes", "List Deployments with desired/ready replicas and image.", K8sNsArgs, k.list_deployments),
        Tool("k8s_list_pods", "kubernetes", "List Pods with phase, readiness, restarts and waiting reason.", K8sNsArgs, k.list_pods),
        Tool("k8s_pod_status", "kubernetes", "Detailed status of one Pod: conditions, container states, restarts, probes, resources.", K8sPodArgs, k.pod_status),
        Tool("k8s_pod_logs", "kubernetes", "Recent logs of a Pod (previous=true for the crashed instance).", K8sLogsArgs, k.pod_logs),
        Tool("k8s_services", "kubernetes", "Inspect Services including ready/not-ready endpoint counts.", K8sNsArgs, k.services),
        Tool("k8s_events", "kubernetes", "Recent (warning) cluster events: probe failures, restarts, scheduling and image errors.", K8sEventsArgs, k.events),
        Tool("k8s_app_version", "kubernetes", "Deployed application version: image tag, labels and ConfigMap model/dataset provenance.", K8sDeploymentArgs, k.app_version),
        Tool("k8s_diagnose_deployment", "kubernetes", "One-shot health diagnosis: unhealthy pods, restarts, warning events, log tails, likely causes.", K8sDeploymentArgs, k.diagnose_deployment),
        Tool("k8s_restart_deployment", "kubernetes", "Rolling-restart a Deployment. WRITE: needs user confirmation; approved namespace only.", K8sDeploymentArgs, k.restart,
             access="write", guard=k.guard_restart, describe=k.describe_restart),
        Tool("k8s_scale_deployment", "kubernetes", "Scale a Deployment. WRITE: needs user confirmation; approved namespace only.", K8sScaleArgs, k.scale,
             access="write", guard=k.guard_scale, describe=k.describe_scale),
    ]
