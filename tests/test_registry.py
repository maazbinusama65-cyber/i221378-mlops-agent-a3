"""Registry behaviour: validation, timeouts, failures, write-confirmation, structured logs."""

from __future__ import annotations

import json
import time

from agent.logging_utils import REDACTED, redact, register_secret
from agent.schemas import Status, ToolArgs
from agent.tools.base import Tool, ToolRegistry, ToolRejected, ToolUnavailable


class Num(ToolArgs):
    x: int


def _tool(fn, **kw) -> Tool:
    return Tool("t", "test", "test tool", Num, fn, **kw)


def test_successful_execution_returns_structured_result(registry: ToolRegistry):
    registry.register(_tool(lambda a: {"double": a.x * 2}))
    res = registry.execute("t", {"x": 4}, "req-1")
    assert res.status is Status.OK and res.data == {"double": 8}
    assert res.request_id == "req-1" and res.evidence_id == "E1" and res.duration_ms >= 0


def test_invalid_arguments_are_rejected_without_running_the_tool(registry: ToolRegistry):
    ran = []
    registry.register(_tool(lambda a: ran.append(a) or {}))
    bad_type = registry.execute("t", {"x": "not-a-number"})
    extra = registry.execute("t", {"x": 1, "surprise": True})
    assert bad_type.status is Status.INVALID_REQUEST and extra.status is Status.INVALID_REQUEST
    assert ran == []


def test_unknown_tool(registry: ToolRegistry):
    assert registry.execute("rm_rf_everything", {}).status is Status.UNKNOWN_TOOL


def test_tool_timeout_is_reported_not_raised(registry: ToolRegistry):
    registry.register(_tool(lambda a: time.sleep(1.5) or {}, timeout_s=0.1))
    res = registry.execute("t", {"x": 1})
    assert res.status is Status.TIMEOUT and "timed out" in res.error


def test_service_failures_are_mapped_to_unavailable_and_error(registry: ToolRegistry):
    def down(_):
        raise ToolUnavailable("MLflow unreachable")

    def boom(_):
        raise RuntimeError("kaboom")

    registry.register(Tool("down", "t", "d", Num, down))
    registry.register(Tool("boom", "t", "d", Num, boom))
    assert registry.execute("down", {"x": 1}).status is Status.UNAVAILABLE
    err = registry.execute("boom", {"x": 1})
    assert err.status is Status.ERROR and "kaboom" in err.error and "Traceback" not in err.error


def test_policy_rejection_is_reported_as_rejected(registry: ToolRegistry):
    def deny(_):
        raise ToolRejected("not approved")

    registry.register(Tool("deny", "t", "d", Num, deny))
    assert registry.execute("deny", {"x": 1}).status is Status.REJECTED


def test_write_tools_never_execute_without_human_confirmation(registry: ToolRegistry):
    ran = []
    registry.register(_tool(lambda a: ran.append(a.x) or {"done": True}, access="write"))
    first = registry.execute("t", {"x": 3})
    assert first.status is Status.CONFIRMATION_REQUIRED and ran == []
    cid = first.data["confirmation_id"]

    assert registry.confirm("deadbeef").status is Status.REJECTED  # bogus token
    done = registry.confirm(cid)
    assert done.status is Status.OK and ran == [3]
    assert registry.confirm(cid).status is Status.REJECTED  # single use


def test_expired_confirmation_is_rejected(registry: ToolRegistry):
    registry.register(_tool(lambda a: {}, access="write"))
    cid = registry.execute("t", {"x": 1}).data["confirmation_id"]
    registry.pending[cid].expires_at = time.time() - 1
    assert registry.confirm(cid).status is Status.REJECTED


def test_structured_logs_have_tool_status_duration_request_id_and_no_secrets(registry: ToolRegistry, settings):
    register_secret("pw-test-123456")
    registry.register(_tool(lambda a: {"ok": 1}))
    registry.execute("t", {"x": 1}, "req-77")
    registry.execute("t", {"x": 1, "password": "pw-test-123456", "token": "abc"}, "req-78")  # invalid but logged
    lines = [json.loads(line) for line in settings.log_file.read_text().splitlines()]
    ok = next(r for r in lines if r["request_id"] == "req-77")
    assert {"tool", "status", "duration_ms", "request_id"} <= ok.keys()
    raw = settings.log_file.read_text()
    assert "pw-test-123456" not in raw and '"abc"' not in raw
    assert REDACTED in raw


def test_redact_scrubs_bearer_tokens_api_keys_and_secret_keys():
    data = {"Authorization": "Bearer abc.def.ghi", "note": "key gsk_ABCDEFGHIJKLMNOP leaked", "nested": {"api_key": "x"}}
    out = redact(data)
    assert out["Authorization"] == REDACTED and out["nested"]["api_key"] == REDACTED
    assert "gsk_ABCDEFGHIJKLMNOP" not in out["note"]
