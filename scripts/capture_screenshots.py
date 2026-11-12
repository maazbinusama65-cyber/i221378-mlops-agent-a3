"""Capture evidence screenshots: MLflow UI, Airflow UI, and terminal output of DVC / Git / kubectl.

Optional developer tool (not needed to run the agent). Uses Playwright driving the locally installed Edge/Chrome:

    pip install playwright
    AIRFLOW_PASSWORD=... KUBECONFIG=... python scripts/capture_screenshots.py
"""

from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
MLFLOW = os.environ.get("MLFLOW_UI", "http://127.0.0.1:5000")
AIRFLOW = os.environ.get("AIRFLOW_BASE_URL", "http://127.0.0.1:8080")

TERMINALS: dict[str, list[list[str]]] = {
    "dvc_provenance": [
        ["git", "tag", "--list", "data-v*", "-n1"],
        ["git", "log", "--oneline", "--decorate", "-8", "--", "data/churn.csv.dvc"],
        ["cat", "data/churn.csv.dvc"],
        ["dvc", "diff", "data-v1", "data-v2"],
        ["dvc", "status"],
    ],
    "kubernetes_state": [
        ["kubectl", "-n", "mlops-agent", "get", "deploy,pods,svc,cm", "-o", "wide"],
        ["kubectl", "-n", "mlops-agent", "get", "events", "--sort-by=.lastTimestamp"],
    ],
}


def terminal_png(page: Page, name: str, commands: list[list[str]]) -> None:
    blocks = []
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False, timeout=120, encoding="utf-8")  # noqa: S603
        shown = " ".join(cmd)
        blocks.append(f'<div class="cmd">$ {html.escape(shown)}</div><pre>{html.escape((proc.stdout + proc.stderr).strip())}</pre>')
    page.set_content(
        "<html><body style='background:#0f1419;color:#d7dde4;font:13px/1.45 Consolas,monospace;padding:20px;width:1500px'>"
        "<style>.cmd{color:#7ee787;margin-top:14px}pre{margin:4px 0 0;white-space:pre-wrap}</style>" + "".join(blocks) + "</body></html>")
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    print("wrote", name)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 900})

        for name, cmds in TERMINALS.items():
            terminal_png(page, name, cmds)

        def shot(url: str, name: str, ctx=None) -> None:
            tab = ctx.new_page() if ctx else browser.new_page(viewport={"width": 1500, "height": 900})
            tab.goto(url, wait_until="load")
            tab.wait_for_timeout(9000)
            tab.screenshot(path=str(OUT / f"{name}.png"))
            tab.close()
            print("wrote", name)

        shot(f"{MLFLOW}/#/experiments/2/runs", "mlflow_experiment_runs")
        shot(f"{MLFLOW}/#/models/churn-classifier", "mlflow_model_registry")

        ctx = browser.new_context(viewport={"width": 1500, "height": 900})
        login = ctx.new_page()
        login.goto(f"{AIRFLOW}/", wait_until="load")
        login.wait_for_timeout(2000)
        login.fill("input[name=username]", os.environ.get("AIRFLOW_USERNAME", "admin"))
        login.fill("input[name=password]", os.environ["AIRFLOW_PASSWORD"])
        login.click("button[type=submit]")
        login.wait_for_timeout(3000)
        login.close()
        shot(f"{AIRFLOW}/dags/churn_training_pipeline/runs", "airflow_dag_runs", ctx)
        browser.close()


if __name__ == "__main__":
    main()
