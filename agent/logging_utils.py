"""Structured JSON logging with secret redaction.

Each tool call is logged as one JSON line: tool name, status, duration, request id - plus a
redacted summary of the arguments. Secrets never reach the log file or the console.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

REDACTED = "***REDACTED***"
_SECRET_KEY_RE = re.compile(r"(password|passwd|secret|token|api[_-]?key|authorization|credential)", re.I)
_PATTERNS = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]*"),  # JWTs
    re.compile(r"(?i)(password|token|api[_-]?key|secret)\s*[=:]\s*\S+"),
]
_registered_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a concrete secret value so it is scrubbed wherever it appears."""
    if value and len(value) >= 6:
        _registered_secrets.add(value)


def redact_text(text: str) -> str:
    for secret in _registered_secrets:
        text = text.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact(value: Any) -> Any:
    """Recursively redact dict keys that look secret and secret-looking strings."""
    if isinstance(value, dict):
        return {k: (REDACTED if _SECRET_KEY_RE.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname}
        payload.update(getattr(record, "fields", {}) or {})
        if "event" not in payload:
            payload["event"] = redact_text(record.getMessage())
        return json.dumps(redact(payload), default=str)


def get_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("mlops-agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    wanted = str(log_file) if log_file else None
    existing = {getattr(h, "_agent_target", None) for h in logger.handlers}
    if wanted and wanted not in existing:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        handler._agent_target = wanted  # type: ignore[attr-defined]
        logger.addHandler(handler)
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, **fields: Any) -> None:
    logger.info(fields.get("event", "event"), extra={"fields": fields})
