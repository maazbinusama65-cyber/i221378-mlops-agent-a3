| Tool | Module | Access | Safety controls | Purpose |
|---|---|---|---|---|
| `dvc_list_datasets` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | List datasets tracked by DVC with md5, size and version label. |
| `dvc_dataset_version` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | Current dataset version, md5 and the Git commit that recorded it. |
| `dvc_resolve_reference` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | Find the Git commit/tag/version for a dataset md5 (from an MLflow run's dvc.md5 tag) or a version label. |
| `dvc_compare_versions` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | Compare two dataset versions (git revs/tags): md5, size, DVC diff. |
| `dvc_check_changes` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | Detect whether the tracked data in the working tree changed vs its recorded md5. |
| `dvc_explain_reproduce` | dvc | read | allowlisted dvc/git sub-commands, approved dataset paths, validated revs | Explain how to reproduce a dataset version; verify=true safely exports it to a sandbox and checks the md5. |
| `mlflow_list_experiments` | mlflow | read | read-only (MlflowClient) | List MLflow experiments with run counts. |
| `mlflow_list_runs` | mlflow | read | read-only (MlflowClient) | List recent runs (newest first) with key metrics and dataset version. |
| `mlflow_get_run` | mlflow | read | read-only (MlflowClient) | Full params, metrics, DVC/Git provenance tags, artifacts and registry versions of one run. |
| `mlflow_best_run` | mlflow | read | read-only (MlflowClient) | Best recent run by a metric, with ranking and production-readiness caveats. |
| `mlflow_compare_runs` | mlflow | read | read-only (MlflowClient) | Compare runs side by side (run_ids or last_n). |
| `mlflow_compare_experiments` | mlflow | read | read-only (MlflowClient) | Compare two or more experiments on a metric. |
| `mlflow_model_registry` | mlflow | read | read-only (MlflowClient) | Registered model versions, aliases (production/candidate) and source runs. |
| `airflow_list_dags` | airflow | read | JWT auth; trigger limited to approved DAGs | List available Airflow DAGs and which are approved to trigger. |
| `airflow_dag_runs` | airflow | read | JWT auth; trigger limited to approved DAGs | Recent DAG runs with state (success/failed/running). |
| `airflow_task_instances` | airflow | read | JWT auth; trigger limited to approved DAGs | Per-task states, tries and durations for a DAG run (latest if run_id omitted). |
| `airflow_failed_tasks` | airflow | read | JWT auth; trigger limited to approved DAGs | Identify failed tasks of the latest failed run, with log excerpts for diagnosis. |
| `airflow_task_logs` | airflow | read | JWT auth; trigger limited to approved DAGs | Fetch the log tail of one task instance. |
| `airflow_trigger_dag` | airflow | **WRITE - human /confirm required** | JWT auth; trigger limited to approved DAGs | Trigger an APPROVED training DAG. Requires user confirmation; never call unless the user asked to run training. |
| `k8s_list_deployments` | kubernetes | read | approved namespaces only; writes bounded + confirmed | List Deployments with desired/ready replicas and image. |
| `k8s_list_pods` | kubernetes | read | approved namespaces only; writes bounded + confirmed | List Pods with phase, readiness, restarts and waiting reason. |
| `k8s_pod_status` | kubernetes | read | approved namespaces only; writes bounded + confirmed | Detailed status of one Pod: conditions, container states, restarts, probes, resources. |
| `k8s_pod_logs` | kubernetes | read | approved namespaces only; writes bounded + confirmed | Recent logs of a Pod (previous=true for the crashed instance). |
| `k8s_services` | kubernetes | read | approved namespaces only; writes bounded + confirmed | Inspect Services including ready/not-ready endpoint counts. |
| `k8s_events` | kubernetes | read | approved namespaces only; writes bounded + confirmed | Recent (warning) cluster events: probe failures, restarts, scheduling and image errors. |
| `k8s_app_version` | kubernetes | read | approved namespaces only; writes bounded + confirmed | Deployed application version: image tag, labels and ConfigMap model/dataset provenance. |
| `k8s_diagnose_deployment` | kubernetes | read | approved namespaces only; writes bounded + confirmed | One-shot health diagnosis: unhealthy pods, restarts, warning events, log tails, likely causes. |
| `k8s_restart_deployment` | kubernetes | **WRITE - human /confirm required** | approved namespaces only; writes bounded + confirmed | Rolling-restart a Deployment. WRITE: needs user confirmation; approved namespace only. |
| `k8s_scale_deployment` | kubernetes | **WRITE - human /confirm required** | approved namespaces only; writes bounded + confirmed | Scale a Deployment. WRITE: needs user confirmation; approved namespace only. |
| `assess_retraining_need` | analysis | read | calls other tools; cannot start retraining | Correlate MLflow + DVC + Kubernetes + Airflow evidence and apply the explicit retraining policy. Never starts retraining. Pass degradation_report ONLY if the user supplied one. |

30 tools total.
