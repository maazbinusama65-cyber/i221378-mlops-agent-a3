"""LLM access with native tool calling (OpenAI-compatible API: Groq, OpenAI, Gemini, OpenRouter, Ollama...)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.logging_utils import register_secret


class LLMError(RuntimeError):
    """The model could not be reached or kept failing after retries."""


class LLMToolCallError(LLMError):
    """The provider rejected the model's tool call (schema validation); the model can be asked to retry."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None  # None => the model produced unparseable JSON
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


@dataclass(frozen=True)
class Preset:
    key_env: str
    base_url: str | None
    model: str


PRESETS: dict[str, Preset] = {
    "groq": Preset("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openai": Preset("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "gemini": Preset("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-flash-latest"),
    "openrouter": Preset("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
}
AUTODETECT = ("groq", "openai", "gemini", "openrouter")


def resolve_config(env: dict[str, str] | None = None) -> dict[str, Any]:
    e = dict(os.environ if env is None else env)
    provider = (e.get("LLM_PROVIDER") or "").lower()
    if not provider:
        provider = next((p for p in AUTODETECT if e.get(PRESETS[p].key_env)), "")
    if not provider:
        raise LLMError("No LLM credentials: set LLM_PROVIDER and the matching key (e.g. GROQ_API_KEY). See .env.example.")
    preset = PRESETS.get(provider, Preset("LLM_API_KEY", None, ""))
    key = e.get("LLM_API_KEY") or e.get(preset.key_env, "")
    base_url = e.get("LLM_BASE_URL") or preset.base_url
    if not key and not (base_url and ("localhost" in base_url or "127.0.0.1" in base_url)):
        raise LLMError(f"API key for {provider!r} is not set ({preset.key_env})")
    return {"provider": provider, "api_key": key or "not-needed", "base_url": base_url, "model": e.get("LLM_MODEL") or preset.model}


class OpenAICompatLLM:
    def __init__(self, api_key: str, model: str, base_url: str | None = None, *, provider: str = "openai",
                 timeout: float = 90.0, max_retries: int = 5, max_tokens: int = 1500) -> None:
        import openai

        register_secret(api_key)
        self.provider, self.model, self.max_retries, self.max_tokens = provider, model, max_retries, max_tokens
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> OpenAICompatLLM:
        cfg = resolve_config(env)
        return cls(cfg["api_key"], cfg["model"], cfg["base_url"], provider=cfg["provider"])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools or None, tool_choice="auto" if tools else None,
                    temperature=0.0, max_tokens=self.max_tokens,
                )
                return self._parse(resp)
            except (self._openai.RateLimitError, self._openai.APIConnectionError, self._openai.InternalServerError) as exc:
                last = exc
                time.sleep(self._wait(exc, attempt))
            except self._openai.APIStatusError as exc:
                if exc.status_code == 400 and "tool call validation failed" in str(exc):
                    raise LLMToolCallError(str(exc)[:400]) from exc
                raise LLMError(f"LLM API error {exc.status_code}: {str(exc)[:300]}") from exc
        raise LLMError(f"LLM unavailable after {self.max_retries} retries: {str(last)[:200]}")

    @staticmethod
    def _wait(exc: Exception, attempt: int) -> float:
        match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s|m)", str(exc))
        if match:
            value = float(match.group(1)) * {"ms": 0.001, "s": 1, "m": 60}[match.group(2)]
            return min(value + 1.0, 60.0)
        return min(2.0 * 2**attempt, 30.0)

    @staticmethod
    def _parse(resp: Any) -> LLMResponse:
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                parsed = json.loads(raw)
                parsed = parsed if isinstance(parsed, dict) else None
            except ValueError:
                parsed = None
            calls.append(ToolCall(tc.id, tc.function.name, parsed, raw))
        usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0), "completion_tokens": getattr(resp.usage, "completion_tokens", 0)} if resp.usage else {}
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)
