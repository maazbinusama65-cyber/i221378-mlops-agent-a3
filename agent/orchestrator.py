"""Tool-calling orchestrator: LLM <-> structured tools, with an evidence ledger and safety rules."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.config import Settings
from agent.llm import LLM, LLMError
from agent.logging_utils import get_logger, log_event
from agent.schemas import Status, ToolResult
from agent.tools import airflow_tools, analysis_tools, dvc_tools, kubernetes_tools, mlflow_tools
from agent.tools.base import ToolRegistry

MAX_TOOL_CHARS = 2600

SYSTEM_PROMPT = """You are an MLOps agent operating real DVC, MLflow, Apache Airflow and Kubernetes systems.
RULES
1. Every operational claim must come from a tool result in THIS conversation; cite the evidence id, e.g. [E3].
   Never answer from memory. If a tool is unavailable, timed out or errored, say the fact is UNKNOWN - never guess or invent values.
2. Pick tools yourself; call several when the question spans systems. Use exact ids returned by earlier tools (run ids, md5, pod names).
3. Never assume the highest metric means production-ready: report the tool's caveats. Dataset versions come from DVC/MLflow provenance only.
4. Never start training or change the cluster on your own. Write tools only return a confirmation request: relay the confirmation id and tell the USER to type /confirm <id>. You cannot confirm.
5. For retraining questions call assess_retraining_need (do not decide from intuition). Recommend retraining only if it reports a trigger.
FINAL ANSWER FORMAT (plain text, concise):
Known facts: bullet list with [E#] citations.
Uncertainties: bullets (unavailable tools, caveats, missing evidence).
Recommended actions: bullets (a recommendation is not an action taken)."""

_CITATION_GROUP = re.compile(r"[\[【]([^\]】]*)[\]】]")


@dataclass
class AgentResponse:
    answer: str
    request_id: str
    steps: int
    evidence: list[ToolResult] = field(default_factory=list)
    pending_confirmations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def build_registry(settings: Settings | None = None) -> ToolRegistry:
    settings = settings or Settings.from_env()
    registry = ToolRegistry(settings)
    registry.register_all(dvc_tools.build_tools(settings))
    registry.register_all(mlflow_tools.build_tools(settings))
    registry.register_all(airflow_tools.build_tools(settings))
    registry.register_all(kubernetes_tools.build_tools(settings))
    registry.register_all(analysis_tools.build_tools(settings, registry))
    return registry


def check_citations(answer: str, ledger: list[ToolResult]) -> list[str]:
    """Return evidence ids cited in the answer that do not exist in this request's ledger."""
    known = {r.evidence_id for r in ledger}
    cited: set[str] = set()
    for group in _CITATION_GROUP.findall(answer.replace("​", "")):
        cited.update(f"E{n}" for n in re.findall(r"E(\d+)", group))
    return sorted(cited - known, key=lambda e: int(e[1:]))


def _compact(result: ToolResult) -> str:
    body: dict[str, Any] = {"evidence_id": result.evidence_id, "tool": result.tool, "status": result.status.value}
    if result.data is not None:
        body["data"] = result.data
    if result.error:
        body["error"] = result.error
    text = json.dumps(body, default=str, separators=(",", ":"))
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + '..."[truncated]"}'
    return text


class Orchestrator:
    def __init__(self, llm: LLM, registry: ToolRegistry, settings: Settings | None = None, max_steps: int = 10) -> None:
        self.llm, self.registry = llm, registry
        self.settings = settings or registry.settings
        self.max_steps = max_steps
        self.logger = get_logger(self.settings.log_file)

    def run(self, user_message: str, history: list[dict[str, Any]] | None = None) -> AgentResponse:
        request_id = uuid.uuid4().hex[:12]
        self.registry.reset_ledger()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": user_message},
        ]
        specs = self.registry.specs()
        pending: list[dict[str, Any]] = []
        log_event(self.logger, event="request_start", request_id=request_id, message=user_message[:200])
        for step in range(1, self.max_steps + 1):
            try:
                reply = self.llm.chat(messages, specs)
            except LLMError as exc:
                log_event(self.logger, event="llm_error", request_id=request_id, error=str(exc))
                return AgentResponse(
                    f"The language model is unavailable ({exc}). No operational conclusions were drawn.",
                    request_id, step, list(self.registry.ledger), pending, error=str(exc),
                )
            if not reply.tool_calls:
                answer = reply.content or "(no answer produced)"
                bogus = check_citations(answer, self.registry.ledger)
                if bogus:
                    log_event(self.logger, event="ungrounded_citation", request_id=request_id, ids=bogus)
                    answer += (
                        "\n\n[GROUNDING WARNING] The answer cites evidence ids that do not exist for this request: "
                        f"{', '.join(bogus)}. Treat the related claims as unverified."
                    )
                log_event(self.logger, event="request_end", request_id=request_id, steps=step, tool_calls=len(self.registry.ledger))
                return AgentResponse(answer, request_id, step, list(self.registry.ledger), pending)
            messages.append({
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or "{}"}}
                    for c in reply.tool_calls
                ],
            })
            for call in reply.tool_calls:
                args = call.arguments if call.arguments is not None else {"__invalid_json__": call.raw_arguments[:80]}
                result = self.registry.execute(call.name, args, request_id)
                if result.status is Status.CONFIRMATION_REQUIRED and result.data:
                    pending.append(result.data)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _compact(result)})
        log_event(self.logger, event="max_steps_reached", request_id=request_id)
        return AgentResponse(
            "Stopped: step limit reached before a final answer. Evidence collected so far is listed below.",
            request_id, self.max_steps, list(self.registry.ledger), pending, error="max_steps",
        )
