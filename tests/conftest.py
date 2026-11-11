from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agent.config import Settings
from agent.llm import LLMResponse, ToolCall
from agent.tools.base import ToolRegistry


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(
        Settings(), repo_root=tmp_path, log_file=tmp_path / "logs" / "agent.jsonl",
        airflow_password="pw-test-123456", tool_timeout_s=1.0,
    )


@pytest.fixture
def registry(settings) -> ToolRegistry:
    return ToolRegistry(settings)


class ScriptedLLM:
    """Deterministic stand-in for the model: replays a script of responses and records what it was shown."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.script.pop(0)


def call(name: str, args: dict | None = None, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(call_id, name, args or {}, "{}")])


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text)
