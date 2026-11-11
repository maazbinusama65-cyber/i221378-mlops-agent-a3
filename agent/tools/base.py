"""Tool registry: validation, timeouts, safety gating, evidence ledger, structured logging."""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.logging_utils import get_logger, log_event, redact, redact_text
from agent.schemas import Status, ToolResult


class ToolUnavailable(RuntimeError):
    """The backing service (MLflow, Airflow, Kubernetes, DVC) cannot be reached."""


class ToolRejected(RuntimeError):
    """A safety policy blocked the request (unapproved namespace / DAG / command)."""


class ToolError(RuntimeError):
    """The operation ran but failed; message is safe to show the agent."""


@dataclass(frozen=True)
class Tool:
    name: str
    module: str  # dvc | mlflow | airflow | kubernetes | analysis
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict[str, Any]]
    access: str = "read"  # "read" | "write"
    guard: Callable[[BaseModel], None] | None = None  # policy check run BEFORE confirmation is offered
    describe: Callable[[BaseModel], str] | None = None  # human text for the confirmation prompt
    timeout_s: float | None = None  # per-tool override of Settings.tool_timeout_s


@dataclass
class PendingAction:
    confirmation_id: str
    tool: str
    args: dict[str, Any]
    summary: str
    expires_at: float


@dataclass
class ToolRegistry:
    settings: Settings
    tools: dict[str, Tool] = field(default_factory=dict)
    ledger: list[ToolResult] = field(default_factory=list)
    pending: dict[str, PendingAction] = field(default_factory=dict)
    _pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=8), repr=False)

    def __post_init__(self) -> None:
        self.logger = get_logger(self.settings.log_file)

    # ------------------------------------------------------------------ registration
    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-style function-calling specs (compact to stay inside small token budgets)."""
        out = []
        for tool in self.tools.values():
            schema = tool.args_model.model_json_schema()
            schema.pop("title", None)
            for prop in schema.get("properties", {}).values():
                prop.pop("title", None)
            out.append({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": schema}})
        return out

    # ------------------------------------------------------------------ execution
    def execute(self, name: str, raw_args: dict[str, Any] | None, request_id: str | None = None, *, confirmed: bool = False) -> ToolResult:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            return self._finish(name, Status.UNKNOWN_TOOL, request_id, started, error=f"unknown tool '{name}'", args=raw_args)
        try:
            args = tool.args_model.model_validate(raw_args or {})
        except ValidationError as exc:
            return self._finish(name, Status.INVALID_REQUEST, request_id, started, error=_short_validation(exc), args=raw_args)

        if tool.access == "write":
            try:
                if tool.guard:
                    tool.guard(args)
            except ToolRejected as exc:
                return self._finish(name, Status.REJECTED, request_id, started, error=str(exc), args=raw_args)
            if not confirmed:
                return self._request_confirmation(tool, args, request_id, started, raw_args)

        timeout = tool.timeout_s or self.settings.tool_timeout_s
        future = self._pool.submit(tool.fn, args)
        try:
            data = future.result(timeout=timeout)
            return self._finish(name, Status.OK, request_id, started, data=data, args=raw_args)
        except FutureTimeout:
            future.cancel()
            return self._finish(name, Status.TIMEOUT, request_id, started, error=f"timed out after {timeout}s", args=raw_args)
        except ToolUnavailable as exc:
            return self._finish(name, Status.UNAVAILABLE, request_id, started, error=str(exc), args=raw_args)
        except ToolRejected as exc:
            return self._finish(name, Status.REJECTED, request_id, started, error=str(exc), args=raw_args)
        except ToolError as exc:
            return self._finish(name, Status.ERROR, request_id, started, error=str(exc), args=raw_args)
        except Exception as exc:  # noqa: BLE001 - the agent must see a structured failure, never a stack trace
            return self._finish(name, Status.ERROR, request_id, started, error=f"{type(exc).__name__}: {exc}", args=raw_args)

    # ------------------------------------------------------------------ confirmations (human only)
    def _request_confirmation(self, tool: Tool, args: BaseModel, request_id: str, started: float, raw_args: Any) -> ToolResult:
        confirmation_id = secrets.token_hex(4)
        summary = tool.describe(args) if tool.describe else f"{tool.name}({args.model_dump()})"
        self.pending[confirmation_id] = PendingAction(
            confirmation_id, tool.name, args.model_dump(), summary, time.time() + self.settings.confirmation_ttl_s
        )
        return self._finish(
            tool.name,
            Status.CONFIRMATION_REQUIRED,
            request_id,
            started,
            data={
                "confirmation_id": confirmation_id,
                "action": summary,
                "instruction": f"Not executed. The USER must type: /confirm {confirmation_id}",
            },
            args=raw_args,
        )

    def confirm(self, confirmation_id: str, request_id: str | None = None) -> ToolResult:
        """Execute a pending write action. Only ever called from the human-facing CLI, never by the LLM."""
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        action = self.pending.pop(confirmation_id, None)
        if action is None or action.expires_at < time.time():
            return self._finish("confirm", Status.REJECTED, request_id, started, error="unknown or expired confirmation id")
        return self.execute(action.tool, action.args, request_id, confirmed=True)

    # ------------------------------------------------------------------ bookkeeping
    def _finish(self, name: str, status: Status, request_id: str, started: float, *, data: dict | None = None,
                error: str | None = None, args: Any = None) -> ToolResult:
        duration = round((time.perf_counter() - started) * 1000, 1)
        result = ToolResult(
            tool=name, status=status, data=data, error=redact_text(error) if error else None,
            duration_ms=duration, request_id=request_id, evidence_id=f"E{len(self.ledger) + 1}",
        )
        self.ledger.append(result)
        log_event(self.logger, event="tool_call", tool=name, status=status.value, duration_ms=duration,
                  request_id=request_id, evidence_id=result.evidence_id, args=redact(args or {}),
                  error=result.error)
        return result

    def reset_ledger(self) -> None:
        self.ledger = []


def _short_validation(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()[:4]]
    return "invalid arguments - " + "; ".join(parts)
