# MLOps AI Agent - DVC + MLflow + Airflow 3 + Kubernetes

Assignment 3 (MLOps & Production AI Engineering, Fall 2026). A conversational agent that answers operational
questions by calling **real tools** against four systems and refuses to state anything those tools did not return.

> "Find the best recent model, identify the dataset version used to train it, check the deployment status, and
> explain whether retraining is necessary." - answered from live MLflow, DVC/Git, Kubernetes and Airflow evidence,
> with known facts, uncertainties and recommended actions kept separate.

* Architecture, safety model, diagrams: [architecture.md](architecture.md)
* **Technical report** (evidence, five tested conversations, two failure investigations): [docs/REPORT.md](docs/REPORT.md)
* Tool catalogue with permissions: [docs/tool-catalogue.md](docs/tool-catalogue.md)

## What is in the box

| Component | Path | Notes |
|---|---|---|
| Agent core (orchestrator, LLM client, policy, CLI) | `agent/` | own tool-calling loop over any OpenAI-compatible API |
| Tool modules (30 tools) | `agent/tools/{dvc,mlflow,airflow,kubernetes,analysis}_tools.py` | structured Pydantic I/O |
| Training code | `training/` | shared by the CLI and the Airflow DAG |
| Airflow DAG | `dags/training_pipeline.py` | validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> register candidate |
| Inference service | `api/main.py`, `Dockerfile` | FastAPI: `/health`, `/ready`, `/version`, `/predict` |
| Kubernetes manifests | `kubernetes/` | 2 replicas, probes, resources, ClusterIP, ConfigMap |
| Local stack | `docker-compose.yml`, `infra/` | MLflow, Airflow 3, k3s |
| Dataset versioning | `data/*.dvc`, `.dvc/` | tags `data-v1`, `data-v2` |
| Tests + CI | `tests/`, `.github/workflows/ci.yml` | 114 tests, 90 % coverage, all services mocked |

## Prerequisites

* Python 3.12, Git, Docker Desktop (Compose v2), `kubectl`
* An API key for an OpenAI-compatible LLM with tool calling (Groq, OpenAI, Gemini, OpenRouter, or a local server)
* ~6 GB free disk for images

## Installation

```bash
git clone https://github.com/maazbinusama65-cyber/i221378-mlops-agent-a3.git
cd i221378-mlops-agent-a3
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Environment configuration

Copy `.env.example` and export the values (never commit real ones). Everything is read from environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER`, `GROQ_API_KEY` (or `OPENAI_API_KEY`...), `LLM_MODEL` | model access | auto-detected from the key |
| `MLFLOW_TRACKING_URI` | MLflow server | `http://127.0.0.1:5000` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow 3 JWT login | `http://127.0.0.1:8080`, `admin`, - |
| `AIRFLOW_ADMIN_PASSWORD` | password the compose Airflow container creates for `admin` | `change-me-local` (local only) |
| `KUBECONFIG` | restricted kubeconfig | - |
| `AGENT_K8S_NAMESPACES` | approved namespaces | `mlops-agent` |
| `AGENT_APPROVED_DAGS` | DAGs the agent may trigger | `churn_training_pipeline` |
| `AGENT_MAX_MODEL_AGE_DAYS` | retraining freshness policy | `30` |
| `AGENT_TOOL_TIMEOUT_S` | per-tool timeout | `30` |

## Launching each service

**One shot** (fresh clone): `bash scripts/bootstrap.sh` runs steps 1-4 below.

1. **MLflow + Airflow 3** - `docker compose up -d --build mlflow airflow`
   MLflow UI: <http://127.0.0.1:5000> - Airflow UI/API: <http://127.0.0.1:8080> (user `admin`, password = `AIRFLOW_ADMIN_PASSWORD`).
2. **DVC data** - `bash scripts/bootstrap_data.sh` regenerates both dataset versions, verifies their md5 against the
   committed pointers (`data-v1`, `data-v2`) and pushes them to the local DVC remote (`.dvc-storage/`).
   Reproduce a version by hand: `git checkout data-v1 -- data/churn.csv.dvc && dvc checkout data/churn.csv.dvc`.
3. **Experiments** - `bash scripts/seed_experiments.sh && python scripts/set_registry_aliases.py && python scripts/sync_deployed_run_id.py`
   trains 10 tracked runs on both dataset versions and exports the serving model.
4. **Kubernetes** - `bash scripts/k8s-up.sh` starts k3s in Docker, builds `churn-api:1.0.0`, loads it into the cluster and
   applies `kubernetes/`. Afterwards `export KUBECONFIG=$PWD/infra/kubeconfig.yaml` (git-ignored).
   Try the service: `kubectl -n mlops-agent port-forward svc/churn-api 8000:80` then `curl localhost:8000/version`.
5. **Training DAG** - auto-discovered from `dags/`; run it through the agent (below) or `airflow dags trigger churn_training_pipeline`.

## Running the agent

```bash
export KUBECONFIG=$PWD/infra/kubeconfig.yaml AIRFLOW_PASSWORD=<admin password> GROQ_API_KEY=<key>
python -m agent.main                      # interactive
python -m agent.main --ask "Which dataset version is tracked by DVC?" --save out.json
```

Example prompts

| Area | Prompt |
|---|---|
| DVC | "Which dataset version trained run `<run id>`? Can I reproduce it?" / "Compare data-v1 and data-v2." / "Has the tracked data changed?" |
| MLflow | "Compare the last five runs and recommend the best candidate by F1." / "What is registered as production?" |
| Airflow | "Trigger the training pipeline and report whether it succeeded." / "Which task failed in the last run and why?" |
| Kubernetes | "Why is my prediction API unavailable?" / "Which version is deployed?" |
| Multi-tool | "Find the best recent model, identify its dataset version, check the deployment, and explain whether retraining is necessary." |

**Write actions need you.** Triggering a DAG, restarting or scaling a Deployment never runs from an LLM request - the
agent replies with a confirmation id and you type `/confirm <id>` (single use, 5 minute TTL). Other chat commands:
`/pending`, `/quit`. Requests outside the approved namespace/DAG list are rejected outright.

Every tool call is logged as one JSON line in `logs/agent.jsonl` (tool, status, duration, request id; secrets redacted).

## Tests and CI

```bash
ruff check .
pytest --cov=agent --cov=training --cov=api --cov-report=term-missing --cov-fail-under=80
```

114 tests cover: tool selection and multi-tool flows, mocked service outages (`unavailable`), timeouts, invalid requests,
unauthorized Kubernetes modifications, confirmation gating, "no invented results when a tool is down", citation grounding,
retraining policy rules, DVC against a real temporary Git+DVC repo, the FastAPI probes, DAG data validation / quality gate,
manifest requirements, and log redaction. GitHub Actions (`.github/workflows/ci.yml`) lints, runs the suite with an
80 % coverage gate, and validates the manifests with kubeconform on every pull request and push; external services are
mocked, so CI needs no secrets or cluster.

## Demonstrating the five scenarios and the failures

```bash
python scripts/run_scenarios.py                 # all (about 20 minutes)
python scripts/run_scenarios.py --only multi    # dataset | model | pipeline | deployment | multi | outage | f1 | f2 | f3
```

Transcripts with the full tool evidence are written to `docs/evidence/conversations/`.

## Security notes

* No credentials are stored in the repo; the compose default `change-me-local` is a placeholder for a local-only Airflow
  container bound to `127.0.0.1`. Use a restricted kubeconfig (namespace-scoped service account) outside the demo.
* `infra/kubeconfig*`, `.env`, DVC cache/remote, and MLflow data are git-ignored.
* Pod logs, tool errors and transcripts are scrubbed of passwords, JWTs and API keys before they reach the model or disk.

See the report for known limitations and proposed improvements.
