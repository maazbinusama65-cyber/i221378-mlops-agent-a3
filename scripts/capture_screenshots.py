"""Capture evidence screenshots: MLflow UI, Airflow UI, and terminal output of DVC / Git / kubectl.

Optional developer tool (not needed to run the agent). Uses Playwright driving the locally installed Edge/Chrome:

    pip install playwright
    AIRFLOW_PASSWORD=... KUBECONFIG=... python scripts/capture_screenshots.py
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
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
        [sys.executable, "-m", "dvc", "diff", "data-v1", "data-v2"],
        [sys.executable, "-m", "dvc", "status"],
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
        shown = " ".join(["python -m dvc" if c == sys.executable else c for c in cmd[:1] + cmd[1:]]).replace("python -m dvc -m dvc", "dvc")
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

        page.goto(f"{MLFLOW}/#/experiments/2", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "mlflow_experiment_runs.png"))
        page.goto(f"{MLFLOW}/#/models/churn-classifier", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "mlflow_model_registry.png"))

        page.goto(f"{AIRFLOW}/", wait_until="networkidle")
        page.fill("input[name=username]", os.environ.get("AIRFLOW_USERNAME", "admin"))
        page.fill("input[name=password]", os.environ["AIRFLOW_PASSWORD"])
        page.click("button[type=submit]")
        page.wait_for_timeout(3000)
        page.goto(f"{AIRFLOW}/dags/churn_training_pipeline/runs", wait_until="networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "airflow_dag_runs.png"))
        page.goto(f"{AIRFLOW}/dags/churn_training_pipeline/graph", wait_until="networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "airflow_dag_graph.png"))
        browser.close()


if __name__ == "__main__":
    main()
