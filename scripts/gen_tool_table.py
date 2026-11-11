"""Print the tool catalogue (name, module, access, description) as Markdown, straight from the registry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.orchestrator import build_registry  # noqa: E402

GUARDS = {
    "dvc": "allowlisted dvc/git sub-commands, approved dataset paths, validated revs",
    "mlflow": "read-only (MlflowClient)",
    "airflow": "JWT auth; trigger limited to approved DAGs",
    "kubernetes": "approved namespaces only; writes bounded + confirmed",
    "analysis": "calls other tools; cannot start retraining",
}


def main() -> None:
    registry = build_registry()
    print("| Tool | Module | Access | Safety controls | Purpose |")
    print("|---|---|---|---|---|")
    for tool in registry.tools.values():
        access = "**WRITE - human /confirm required**" if tool.access == "write" else "read"
        print(f"| `{tool.name}` | {tool.module} | {access} | {GUARDS[tool.module]} | {tool.description} |")
    print(f"\n{len(registry.tools)} tools total.")


if __name__ == "__main__":
    main()
