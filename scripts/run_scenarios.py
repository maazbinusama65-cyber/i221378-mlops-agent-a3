"""Run the five demonstration scenarios and the injected-failure investigations against the LIVE stack.

    python scripts/run_scenarios.py [--only NAME ...] [--out docs/evidence/conversations]

Requires the docker compose stack, the k3s cluster, and an LLM key in the environment.
Operator (human) actions - confirming a write, injecting/repairing a fault - are performed by this script and
clearly labelled in the transcript; the agent itself never confirms or modifies anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.llm import OpenAICompatLLM  # noqa: E402
from agent.logging_utils import redact  # noqa: E402
from agent.orchestrator import AgentResponse, Orchestrator, build_registry  # noqa: E402

NS = "mlops-agent"
DEPLOYED_RUN = "9750c8125dc4409089d4cf1522fd1d72"


def kubectl(*args: str) -> str:
    out = subprocess.run(["kubectl", *args], capture_output=True, text=True, check=False, timeout=120)  # noqa: S603, S607
    return (out.stdout + out.stderr).strip()


class Runner:
    def __init__(self, out_dir: Path) -> None:
        self.settings = Settings.from_env()
        self.registry = build_registry(self.settings)
        self.agent = Orchestrator(OpenAICompatLLM.from_env(), self.registry, self.settings)
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def ask(self, name: str, question: str, *, operator_notes: list[str] | None = None) -> AgentResponse:
        print(f"\n=== {name} ===\nQ: {question}")
        resp = self.agent.run(question)
        print(resp.answer[:1500])
        payload = {
            "scenario": name, "question": question, "request_id": resp.request_id, "llm_steps": resp.steps,
            "answer": resp.answer, "pending_confirmations": resp.pending_confirmations,
            "operator_notes": operator_notes or [],
            "tool_evidence": [r.model_dump(mode="json") for r in resp.evidence],
        }
        (self.out_dir / f"{name}.json").write_text(json.dumps(redact(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        return resp

    def operator_confirm(self, resp: AgentResponse) -> dict:
        """The human typing /confirm <id> in the CLI."""
        cid = resp.pending_confirmations[0]["confirmation_id"]
        result = self.registry.confirm(cid)
        print(f"[operator] /confirm {cid} -> {result.status.value}")
        return result.model_dump(mode="json")

    def wait_for_run(self, timeout: int = 420) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            runs = self.registry.execute("airflow_dag_runs", {"limit": 1}).data["runs"]
            if runs and runs[0]["state"] in ("success", "failed"):
                return runs[0]["state"]
            time.sleep(10)
        return "timeout"

    def trigger_as_operator(self, conf: dict) -> str:
        pending = self.registry.execute("airflow_trigger_dag", {"conf": conf})
        self.registry.confirm(pending.data["confirmation_id"])
        time.sleep(3)
        return self.wait_for_run()


# --------------------------------------------------------------------------------- scenarios
def dataset_provenance(r: Runner) -> None:
    r.ask("01_dataset_provenance",
          f"Which dataset version and git commit were used to train MLflow run {DEPLOYED_RUN}? "
          "Verify that this exact dataset can be reproduced and tell me how.")


def model_selection(r: Runner) -> None:
    r.ask("02_model_selection", "Compare the last five training runs and recommend the best candidate based on F1 score.")


def pipeline_execution(r: Runner) -> None:
    resp = r.ask("03a_pipeline_trigger_request", "Trigger the model training pipeline and report whether execution succeeded.")
    assert resp.pending_confirmations, "agent must request confirmation, not trigger on its own"
    confirm = r.operator_confirm(resp)
    state = r.wait_for_run()
    r.ask("03b_pipeline_result", "What is the status of the most recent training pipeline run? List every task's state.",
          operator_notes=[f"operator typed /confirm {resp.pending_confirmations[0]['confirmation_id']}: {confirm['status']}",
                          f"run finished with state={state}"])


def deployment_diagnosis(r: Runner) -> None:
    r.ask("04_deployment_health", "Is my prediction API deployment healthy? Which application version is deployed and what do the services expose?")


def multi_tool(r: Runner) -> None:
    r.ask("05_multi_tool_reasoning",
          "Find the best recent model, identify the dataset version used to train it, check the deployment status, "
          "and explain whether retraining is necessary.")


def mlflow_outage(r: Runner) -> None:
    dead = dataclasses.replace(r.settings, mlflow_uri="http://127.0.0.1:9", tool_timeout_s=15.0)
    saved, r.agent = r.agent, Orchestrator(OpenAICompatLLM.from_env(), build_registry(dead), dead)
    try:
        r.ask("06_tool_unavailable_mlflow_down", "What is the best model by F1 in MLflow, and what was its F1 score?")
    finally:
        r.agent = saved


def failure_airflow_validation(r: Runner) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_corrupt_dataset.py")], check=True)  # noqa: S603
    state = r.trigger_as_operator({"data_path": "data/failure_injection/churn_corrupt.csv"})
    r.ask("F1_airflow_data_validation_failure", "The last training pipeline run did not succeed. Which task failed, why, "
          "and did anything downstream get promoted?", operator_notes=["injected: DAG run with a corrupted dataset (null tenure values)", f"run state={state}"])


def failure_airflow_gate(r: Runner) -> None:
    state = r.trigger_as_operator({"min_f1": 0.99})
    r.ask("F2_airflow_quality_gate_failure", "The latest training run failed. Which task failed, what were the numbers, "
          "and was a model registered or promoted?", operator_notes=["injected: quality gate min_f1=0.99 (unreachable)", f"run state={state}"])


def failure_k8s_bad_config(r: Runner) -> None:
    notes = ["injected: ConfigMap MODEL_PATH -> /app/models/model_v2.joblib (does not exist), then pods deleted so they re-read it"]
    notes.append(kubectl("-n", NS, "patch", "configmap", "churn-api-config", "--type", "merge",
                         "-p", '{"data":{"MODEL_PATH":"/app/models/model_v2.joblib"}}'))
    notes.append(kubectl("-n", NS, "delete", "pods", "-l", "app=churn-api", "--wait=false"))
    time.sleep(50)
    diag = r.ask("F3_k8s_prediction_api_unavailable", "Why is my prediction API unavailable? Investigate the pods, probes, events and logs.",
                 operator_notes=notes)
    # Repair: the operator restores the ConfigMap; the agent proposes a restart that the human confirms.
    repair = [kubectl("apply", "-f", str(ROOT / "kubernetes" / "configmap.yaml"))]
    fix = r.ask("F3b_k8s_restart_request", "The ConfigMap has been corrected. Restart the churn-api deployment so it picks up the fix.",
                operator_notes=repair)
    if fix.pending_confirmations:
        repair.append(json.dumps(r.operator_confirm(fix)["data"]))
    time.sleep(40)
    r.ask("F3c_k8s_recovery_check", "Is the prediction API healthy again?", operator_notes=repair)
    _ = diag


SCENARIOS = {
    "dataset": dataset_provenance, "model": model_selection, "pipeline": pipeline_execution,
    "deployment": deployment_diagnosis, "multi": multi_tool, "outage": mlflow_outage,
    "f1": failure_airflow_validation, "f2": failure_airflow_gate, "f3": failure_k8s_bad_config,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "evidence" / "conversations")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runner = Runner(args.out)
    for name in args.only or list(SCENARIOS):
        SCENARIOS[name](runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
