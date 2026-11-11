"""Command-line interface.

    python -m agent.main                       interactive chat
    python -m agent.main --ask "..."           single question
    python -m agent.main --ask "..." --save docs/evidence/conversations/x.json

In chat: ``/confirm <id>`` executes a pending write action (human only), ``/pending`` lists them, ``/quit`` exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import Settings
from agent.llm import LLMError, OpenAICompatLLM
from agent.logging_utils import redact
from agent.orchestrator import AgentResponse, Orchestrator, build_registry


def _print_response(resp: AgentResponse) -> None:
    print("\n" + resp.answer)
    if resp.pending_confirmations:
        print("\nPending confirmations:")
        for p in resp.pending_confirmations:
            print(f"  /confirm {p['confirmation_id']}   -> {p['action']}")
    print(f"\n--- evidence (request {resp.request_id}, {resp.steps} LLM step(s)) ---")
    for r in resp.evidence:
        print(f"  {r.evidence_id:<4} {r.tool:<28} {r.status.value:<22} {r.duration_ms:>8.1f} ms")


def _save(path: Path, question: str, resp: AgentResponse) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": question, "request_id": resp.request_id, "llm_steps": resp.steps, "answer": resp.answer,
        "pending_confirmations": resp.pending_confirmations,
        "tool_evidence": [r.model_dump(mode="json") for r in resp.evidence],
    }
    path.write_text(json.dumps(redact(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved transcript -> {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLOps AI agent (DVC + MLflow + Airflow + Kubernetes)")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--save", type=Path, help="write the full evidence transcript (JSON) here")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = Settings.from_env()
    try:
        llm = OpenAICompatLLM.from_env()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    registry = build_registry(settings)
    agent = Orchestrator(llm, registry, settings)
    print(f"MLOps agent ready - model={llm.model} provider={llm.provider} tools={len(registry.tools)}")

    if args.ask:
        resp = agent.run(args.ask)
        _print_response(resp)
        if args.save:
            _save(args.save, args.ask, resp)
        return 0 if resp.error is None else 1

    history: list[dict] = []
    print("Type a question. /pending, /confirm <id>, /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/pending":
            for pid, act in registry.pending.items():
                print(f"  /confirm {pid}  -> {act.summary}")
            continue
        if line.startswith("/confirm"):
            parts = line.split()
            result = registry.confirm(parts[1]) if len(parts) == 2 else None
            print(result.model_dump_json(indent=2) if result else "usage: /confirm <id>")
            continue
        resp = agent.run(line, history)
        history = (history + [{"role": "user", "content": line}, {"role": "assistant", "content": resp.answer}])[-8:]
        _print_response(resp)
        if args.save:
            _save(args.save, line, resp)


if __name__ == "__main__":
    raise SystemExit(main())
