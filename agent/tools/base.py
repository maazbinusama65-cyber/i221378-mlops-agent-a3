"""Tool registry: validation, timeouts, safety gating, evidence ledger, structured logging."""

from __future__ import annotations

import contextvars
import secrets
import threading
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

_epoch_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("ledger_epoch", default=None)
MAX_PENDING = 20


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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _epoch: int = 0
    _seq: int = 0

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
        # Bind this call (and any nested calls made from its worker thread) to the current ledger epoch so a
        # late result from a timed-out call can never pollute a later request's evidence ledger.
        token = _epoch_var.set(_epoch_var.get() if _epoch_var.get() is not None else self._epoch)
        try:
            return self._execute(name, raw_args, request_id, confirmed)
        finally:
            _epoch_var.reset(token)

    def _execute(self, name: str, raw_args: dict[str, Any] | None, request_id: str, confirmed: bool) -> ToolResult:
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
        future = self._pool.submit(contextvars.copy_context().run, tool.fn, args)
        try:
            data = future.result(timeout=timeout)
            return self._finish(name, Status.OK, request_id, started, data=data, args=raw_args)
        except FutureTimeout:
            future.cancel()
            note = f"timed out after {timeout}s"
            if tool.access == "write":
                note += "; OUTCOME UNKNOWN - the change may still have been applied. Verify the current state before retrying."
            return self._finish(name, Status.TIMEOUT, request_id, started, error=note, args=raw_args)
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
        now = time.time()
        for stale in [cid for cid, p in self.pending.items() if p.expires_at < now]:
            del self.pending[stale]
        if len(self.pending) >= MAX_PENDING:
            return self._finish(tool.name, Status.REJECTED, request_id, started,
                                error=f"too many unconfirmed actions pending ({MAX_PENDING}); confirm or wait for expiry", args=raw_args)
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
        epoch = _epoch_var.get()
        with self._lock:
            current = epoch is None or epoch == self._epoch
            evidence_id = None
            if current:
                self._seq += 1
                evidence_id = f"E{self._seq}"
        result = ToolResult(
            tool=name, status=status, data=redact(data) if data is not None else None,
            error=redact_text(error) if error else None, duration_ms=duration, request_id=request_id, evidence_id=evidence_id,
        )
        if current:
            with self._lock:
                self.ledger.append(result)
        log_event(self.logger, event="tool_call", tool=name, status=status.value, duration_ms=duration,
                  request_id=request_id, evidence_id=evidence_id, stale=not current, args=redact(args or {}), error=result.error)
        return result

    def reset_ledger(self) -> None:
        with self._lock:
            self._epoch += 1
            self._seq = 0
            self.ledger = []


def _short_validation(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()[:4]]
    return "invalid arguments - " + "; ".join(parts)
