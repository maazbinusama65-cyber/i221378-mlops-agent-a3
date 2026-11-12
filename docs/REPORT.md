# Technical Report - MLOps AI Agent (Assignment 3)

Student: i221378 - Repository: `maazbinusama65-cyber/i221378-mlops-agent-a3`

All evidence below was produced by the code in this repository against a **live** local stack: MLflow 3.18 server,
Apache Airflow 3.1 (REST API v2, JWT), a k3s Kubernetes cluster (v1.31) running the FastAPI inference service, and a DVC
repository with a local remote. The LLM was Groq `openai/gpt-oss-120b` through native tool calling. Raw transcripts
(every tool call with status, duration, request id, evidence id) are in [`docs/evidence/conversations/`](evidence/conversations/);
raw tool outputs are in [`docs/evidence/exports/`](evidence/exports/); screenshots are in [`screenshots/`](../screenshots/).

## 1. Complete system architecture

Full diagrams and design rationale: [`architecture.md`](../architecture.md). In one paragraph: a CLI sends the user request
to an **Orchestrator**, which runs a tool-calling loop with an LLM. Every call goes through a **Tool Registry** that
validates inputs (Pydantic, `extra="forbid"`), enforces allowlists, applies per-tool timeouts, gates write actions behind a
human `/confirm`, redacts secrets, writes one structured JSON log line per call and records the result in a numbered
**evidence ledger** (`E1..En`). The LLM only ever sees tool results; the final answer must cite them, and citations to
non-existent evidence ids are flagged. 30 tools in four modules plus one cross-module correlator
(`assess_retraining_need`) that applies an explicit retraining policy.

```text
User/CLI -> Orchestrator <-> LLM
               |
          Tool Registry  (validate - allowlist - timeout - confirm gate - redact - log - ledger)
     +---------+----------+-----------+
   DVC+Git   MLflow     Airflow 3    Kubernetes        + assess_retraining_need (uses all four)
```

Environment: `docker-compose.yml` (MLflow, Airflow, k3s), `scripts/bootstrap.sh`, DVC remote, `kubernetes/` manifests,
`Dockerfile` for the inference image. CI: `.github/workflows/ci.yml`.

## 2. Tool definitions and permissions

Generated from the live registry (`scripts/gen_tool_table.py`); also in [`tool-catalogue.md`](tool-catalogue.md).

| Module | Tools | Access | Controls |
|---|---|---|---|
| DVC (6) | `dvc_list_datasets`, `dvc_dataset_version`, `dvc_resolve_reference`, `dvc_compare_versions`, `dvc_check_changes`, `dvc_explain_reproduce` | read; `verify=true` exports into `.agent-sandbox/` only | subprocess argument lists (no shell); allowlisted `dvc` (`status diff get list data version`) and read-only `git` sub-commands; dataset path allowlist; revisions/md5 validated with `fullmatch` against option injection |
| MLflow (7) | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_get_run`, `mlflow_best_run`, `mlflow_compare_runs`, `mlflow_compare_experiments`, `mlflow_model_registry` | read-only | `MlflowClient` only; fail-fast retries; unreachable server -> `unavailable` |
| Airflow (6) | `airflow_list_dags`, `airflow_dag_runs`, `airflow_task_instances`, `airflow_failed_tasks`, `airflow_task_logs`, `airflow_trigger_dag` | 5 read, **1 write** | JWT login with refresh; ids restricted to `[A-Za-z0-9_.:+-]` (no path smuggling); trigger limited to approved DAGs, allowlisted/bounded `conf` keys, **human `/confirm` required** |
| Kubernetes (10) | `k8s_list_deployments`, `k8s_list_pods`, `k8s_pod_status`, `k8s_pod_logs`, `k8s_services`, `k8s_events`, `k8s_app_version`, `k8s_diagnose_deployment`, **`k8s_restart_deployment`, `k8s_scale_deployment`** | 8 read, **2 write** | every call (reads too) limited to approved namespaces (`mlops-agent`); scale <= 5 replicas; writes need **human `/confirm`**; pod logs redacted |
| Analysis (1) | `assess_retraining_need` | read | calls the other tools; cannot start training; policy in `agent/policy.py` |

The LLM has **no** `confirm` tool. A write request returns `confirmation_required` and a single-use id (5 min TTL, max 20
pending); only the human CLI command `/confirm <id>` executes it. A timed-out write reports "outcome unknown - verify
state before retrying".

**Retraining policy** (explicit, deterministic): recommend retraining only if (a) the deployed model is older than the
freshness limit (30 d), (b) it was trained on an older DVC dataset version than the current one, (c) the latest Airflow run
failed its quality gate, or (d) the user supplied degradation evidence. Anything else, including "a better F1 exists",
is not a trigger. Missing evidence gives `insufficient_evidence`, never a guess.

## 3. Evidence from DVC, MLflow, Airflow and Kubernetes

| System | Evidence |
|---|---|
| DVC | [`screenshots/dvc_provenance.png`](../screenshots/dvc_provenance.png): tags `data-v1`/`data-v2`, git history of `churn.csv.dvc`, recorded md5 + `meta.version`, `dvc diff data-v1 data-v2`, clean `dvc status`. Raw: `exports/dvc_dataset_version.json`, `dvc_compare_versions.json`, `dvc_explain_reproduce.json` (verify=true: exported `data-v1` to a sandbox, **md5 `b776848b...` matches the recorded value**, working tree untouched) |
| MLflow | [`screenshots/mlflow_experiment_runs.png`](../screenshots/mlflow_experiment_runs.png) (runs with Airflow-triggered and CLI runs), [`mlflow_model_registry.png`](../screenshots/mlflow_model_registry.png) (v1 = `production`, v4 = `candidate`). Raw: `exports/mlflow_best_run.json`, `mlflow_compare_runs.json`, `mlflow_model_registry.json` |
| Airflow | [`screenshots/airflow_dag_runs.png`](../screenshots/airflow_dag_runs.png): one successful run and two failed runs with per-task grid (failed validation, failed gate, `upstream_failed` downstream). Raw: `exports/airflow_dag_runs.json`, `airflow_task_instances.json` |
| Kubernetes | [`screenshots/kubernetes_state.png`](../screenshots/kubernetes_state.png): `deploy,pods,svc,cm` (2/2 ready, ClusterIP, ConfigMap) and the event stream including the injected failure and recovery. Raw: `exports/k8s_diagnose_deployment.json`, `k8s_app_version.json`, `k8s_services.json` |

Dataset provenance chain (every MLflow run carries `dvc.dataset_version`, `dvc.md5`, `dvc.dirty`, `git.commit`; the
Kubernetes ConfigMap carries `MODEL_RUN_ID` and `DATASET_VERSION`; the agent joins them).

## 4. Five tested agent conversations (actual tool evidence)

Each transcript JSON contains the question, the answer, and every tool result with `request_id`, `status`, `duration_ms`.

| # | Scenario | Question (abridged) | Tools actually called | Outcome |
|---|---|---|---|---|
| 1 | Dataset provenance - [`01_dataset_provenance.json`](evidence/conversations/01_dataset_provenance.json) | Which dataset version and git commit trained run `9750c812...`? Can it be reproduced? | `mlflow_get_run`, `dvc_explain_reproduce` (first call errored: it passed the label `v1` as a git rev, `git show failed: invalid object name 'v1'`; the agent recovered via `dvc_resolve_reference` -> tag `data-v1`), `dvc_resolve_reference`, `dvc_explain_reproduce(verify)` | Dataset **v1**, md5 `b776848b...`, recorded in commit `70810ce` (tag `data-v1`); code commit `ae7287e`. Reproduction **verified**: sandbox export md5 equals the recorded md5. Reproduction steps given. |
| 2 | Model selection - [`02_model_selection.json`](evidence/conversations/02_model_selection.json) | Compare the last five runs; recommend the best by F1 | `mlflow_compare_runs` (1 call) | Best F1 = 0.8239 (`v2-rf-overfit`) **but flagged**: train F1 = 1.0 (gap 0.18) and margin over the runner-up < 0.02, so "production readiness is not established". It did not simply crown the top F1. |
| 3 | Pipeline execution - [`03a`](evidence/conversations/03a_pipeline_trigger_request.json), [`03b`](evidence/conversations/03b_pipeline_result.json) | Trigger the training pipeline and report whether it succeeded | `airflow_trigger_dag` -> `confirmation_required`; operator typed `/confirm 8a252f93`; then `airflow_dag_runs`, `airflow_task_instances` | Real DAG run `success`; all six tasks `success`. The agent could not start the run on its own. |
| 4 | Deployment diagnosis - [`04_deployment_health.json`](evidence/conversations/04_deployment_health.json) | Is the prediction API healthy? Which version? | `k8s_diagnose_deployment`, `k8s_app_version`, `k8s_services` | 2/2 ready, image `churn-api:1.0.0`, ConfigMap model run `9750c812...` on dataset v1, ClusterIP with 2 ready endpoints. |
| 5 | Multi-tool reasoning - [`05_multi_tool_reasoning.json`](evidence/conversations/05_multi_tool_reasoning.json) | The mandatory scenario: best recent model, its dataset, deployment status, is retraining needed? | 11 tool calls across all four systems incl. `mlflow_best_run`, `dvc_resolve_reference`, `k8s_app_version`, `k8s_diagnose_deployment`, `airflow_dag_runs`, `assess_retraining_need` | Facts, uncertainties and actions kept separate. Verdict **`retraining_recommended`** solely because of the data-freshness trigger (deployed model trained on v1, current dataset v2). The best run's overfitting caveat is reported. **No training was started.** |

Additional conversation - tool unavailable ([`06_tool_unavailable_mlflow_down.json`](evidence/conversations/06_tool_unavailable_mlflow_down.json)):
MLflow pointed at a dead port. Both tools returned `unavailable`; the agent answered "Known facts: none", stated the best
model and F1 **cannot be determined**, and recommended restoring the server. No metric was invented.

## 5. Failure investigations (root cause and resolution)

### Failure A - Kubernetes: "Why is my prediction API unavailable?" ([`F3_...`](evidence/conversations/F3_k8s_prediction_api_unavailable.json), [`F3b`](evidence/conversations/F3b_k8s_restart_request.json), [`F3c`](evidence/conversations/F3c_k8s_recovery_check.json))

*Injected fault:* the ConfigMap `MODEL_PATH` was changed to `/app/models/model_v2.joblib` (a file that is not in the image) and the pods
were deleted so they re-read it (a realistic "config pushed, pods rescheduled" scenario).

*Symptoms collected by the agent (one `k8s_diagnose_deployment` call + follow-ups):* desired 2, **ready 0**; both pods `Running` but
**0/1 Ready** (liveness passes, readiness fails with HTTP 503 -> Service has no ready endpoints); warning events
"Readiness probe failed ... statuscode: 503"; pod logs
`WARNING churn-api readiness check failed: model not loaded (FileNotFoundError: [Errno 2] No such file or directory: '/app/models/model_v2.joblib')`.

*Root cause (from evidence):* configuration drift - `MODEL_PATH` points at a model file that does not exist in the image, so the
model never loads and the readiness probe keeps the pods out of rotation. Liveness is deliberately independent of model loading,
which is why this is **not** a restart loop - the agent distinguished "Running but not Ready" from CrashLoopBackOff.

*Resolution:* the operator restored the ConfigMap (`kubectl apply -f kubernetes/configmap.yaml`); the agent was asked to restart
the deployment and replied with a confirmation request (`/confirm f546baaa`); the operator confirmed, the rolling restart ran
inside the approved namespace, and the recovery check reported **2/2 ready, healthy = true** (the agent still mentioned the
probe-failure events from the previous 15 minutes - correct, they are from the failure window).

### Failure B - Airflow: data-validation failure ([`F1_...`](evidence/conversations/F1_airflow_data_validation_failure.json))

*Injected fault:* a DAG run on a corrupted dataset copy with ~11 % of `tenure_months` missing (as after a broken upstream export).

*Agent diagnosis (`airflow_dag_runs`, `airflow_failed_tasks`, `mlflow_model_registry`):* run state `failed`; root-cause task
`validate_data`, which raised `AirflowFailException` because of **334 null values** in feature/target columns; all five downstream
tasks are `upstream_failed` and never ran; the registry shows no new candidate was created by the failed run (the latest `candidate`
came from the previous successful run).

*Root cause and resolution:* bad input data, caught at the first stage exactly as designed - no retries were burned
(`AirflowFailException` is non-retryable) and nothing was promoted. Resolution is upstream: fix/replace the data and re-run.

### Additional investigations

* **Quality-gate failure** ([`F2_...`](evidence/conversations/F2_airflow_quality_gate_failure.json)): triggering with `min_f1=0.99`. The agent found the failed task `quality_gate` with the log line
  `f1 0.820 < min_f1 0.99`, `register_candidate = upstream_failed`, and confirmed from MLflow that the run's `registered_versions` is empty - a failed gate cannot promote a model.
* **Unexpected real bug** ([`real-failure-register-candidate.json`](evidence/real-failure-register-candidate.json)): the very first DAG run failed in `register_candidate` after the retry policy was exhausted (3 tries). The agent's `airflow_failed_tasks`
  initially returned only "Task failed with exception"; I extended the log parser to surface `error_detail`, which exposed
  `ValueError: The key 'run_id' in args is a part of kwargs and therefore reserved`. Root cause: my task parameter was named `run_id`, colliding with Airflow's reserved context key. Fix: renamed to `candidate_run_id`; the next runs succeeded.
* **Infrastructure incident:** during the scenario run the Airflow scheduler and DAG processor died (`BrokenPipeError` from `airflow standalone`'s multiplexed output pipe) and a run hung in `running`. Detected via `/monitor/health`; fix: replaced `standalone` with a plain process supervisor (each component logs straight to the container; container restarts if one exits).
* **Misleading diagnosis found in testing:** the agent initially read hour-old startup probe warnings as a current fault. Fix: diagnosis only treats events from the last 15 minutes as current (regression test added).
* **Provider-side tool-call rejection:** Groq rejected a call with `conf: null`; the schema was made nullable and the orchestrator now feeds provider validation errors back to the model for up to two repairs.

## 6. Testing, reliability and safety

`pytest`: **114 tests, 90 % coverage** (CI gate 80 %); all external systems are mocked, plus DVC tests that run against a real
temporary Git+DVC repository. Covered requirements:

| Requirement | Test(s) |
|---|---|
| >= 8 meaningful tests | 114 across 11 files |
| Mocked tool responses for unavailable services | `test_kubernetes_tools`, `test_airflow_tools` (httpx MockTransport), `test_mlflow_tools` (fake client) |
| Successful tool selection and execution | `test_orchestrator::test_agent_selects_and_executes_the_requested_tool`, multi-tool chain test |
| Timeouts, invalid requests, service failures | `test_registry` (timeout -> `timeout`, bad args -> `invalid_request`, outage -> `unavailable`), orchestrator malformed-JSON test |
| Agent does not invent results when a tool is down | `test_agent_does_not_invent_results_when_a_tool_is_unavailable`, `test_correlator_reports_insufficient_evidence_when_every_system_is_down`, `test_missing_evidence_yields_insufficient_evidence_not_guesses` |
| Unauthorized Kubernetes modifications rejected | `test_modification_in_unapproved_namespace_is_rejected_and_never_reaches_the_cluster`, `test_unauthorized_cluster_modification_via_the_agent_is_rejected`, replica-limit test, "LLM cannot confirm its own write" |
| Structured logs without secrets | `test_structured_logs_have_tool_status_duration_request_id_and_no_secrets`, redaction tests for JWT/Basic/URL credentials/API keys |

Structured log line (from `logs/agent.jsonl`): `{"ts": "...", "event": "tool_call", "tool": "k8s_diagnose_deployment", "status": "ok", "duration_ms": 111.5, "request_id": "...", "evidence_id": "E6", "args": {...}}`.

An independent code/security review found no path around the confirmation gate or the allowlists; its medium findings were fixed with
regression tests (`tests/test_hardening.py`): path-segment validation, restricted DAG `conf`, redaction of all tool output, "outcome unknown"
for timed-out writes, an epoch-safe evidence ledger, bounded pending confirmations.

## 7. Known limitations and proposed improvements

| Limitation | Improvement |
|---|---|
| LLM behaviour varies run to run: answers occasionally include unsupported suggestions (e.g. proposing `model_v1.joblib` in Failure A, marked "[no evidence]") | Add a verifier pass that checks every claim sentence against the ledger; evaluation harness with golden transcripts |
| Citation check validates that ids exist, not that they support the claim | Claim-level entailment check |
| Local single-node k3s, local DVC remote, SQLite-backed Airflow, synthetic dataset | Managed cluster (namespace-scoped ServiceAccount + RBAC), S3/GCS DVC remote, Postgres + CeleryExecutor/KubernetesExecutor, real data |
| Pending confirmations are in memory and per CLI process; the CLI user is implicitly the approver | Persist pending actions, authenticate approvers, add an audit trail and two-person approval for production |
| Retraining policy thresholds are static (30 days, dataset version) | Policy-as-config, drift metrics from serving logs, shadow-traffic degradation signals as a first-class input |
| Airflow durations can look negative after a host clock jump (the agent reported this honestly in conversation 3b) | NTP-safe timestamps / monotonic durations in monitoring |
| Token limits of the hosted model constrain tool-result size (outputs are compacted and truncated at 2.6 k chars) | Result summarisation / retrieval over the ledger instead of inlining |
| DVC tools assume one approved dataset path and a local remote for `in_remote` checks | Multi-dataset config and remote-aware status via `dvc status -c` |
| The DAG trains in-process in the Airflow worker | KubernetesPodOperator with the training image and resource limits |
