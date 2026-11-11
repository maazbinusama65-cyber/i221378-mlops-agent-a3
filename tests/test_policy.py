"""Retraining policy: only explicit, evidence-backed triggers recommend retraining."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent.policy import Evidence, assess

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def evidence(*, deployed_ds="v2", current_ds="v2", run_age_days=3, runs=("success",), failed_root=None, degraded=None) -> Evidence:
    started = (NOW - timedelta(days=run_age_days)).isoformat()
    ev = Evidence(
        deployment={"version_label": "1.0.0", "image": "churn-api:1.0.0",
                    "config": {"DATASET_VERSION": deployed_ds, "MODEL_RUN_ID": "r1"}},
        deployment_health={"ready": 2, "desired": 2, "healthy": True},
        deployed_run={"name": "deployed", "started": started, "metrics": {"f1": 0.8}, "provenance": {"dvc.dataset_version": deployed_ds}},
        best_run={"metric": "f1", "best": {"name": "top", "metrics": {"f1": 0.95}, "dataset_version": current_ds}, "caveats": []},
        current_dataset={"dataset_version": current_ds, "md5": "a" * 32},
        airflow_runs={"runs": [{"dag_run_id": f"run{i}", "state": s} for i, s in enumerate(runs)]},
        degradation_report=degraded,
    )
    if failed_root:
        ev.airflow_failed = {"failed_run": {"dag_run_id": "run0"}, "root_cause_task": failed_root}
    return ev


def verdict(ev: Evidence, **kw):
    return assess(ev, max_model_age_days=30, now=NOW, **kw)


def test_better_model_alone_is_not_a_reason_to_retrain():
    out = verdict(evidence())
    assert out["verdict"] == "retraining_not_required" and out["triggers_met"] == []
    assert out["retraining_started"] is False


def test_stale_dataset_version_triggers_data_freshness_policy():
    out = verdict(evidence(deployed_ds="v1", current_ds="v2"))
    assert out["verdict"] == "retraining_recommended"
    assert any("Data freshness" in t for t in out["triggers_met"])


def test_model_age_beyond_policy_triggers_freshness_policy():
    out = verdict(evidence(run_age_days=45))
    assert any("Freshness policy" in t and "45" in t for t in out["triggers_met"])


def test_failed_quality_gate_in_latest_run_triggers():
    out = verdict(evidence(runs=("failed", "success"), failed_root="quality_gate"))
    assert any("quality gate" in t for t in out["triggers_met"])


def test_failed_validation_is_not_a_model_quality_trigger():
    out = verdict(evidence(runs=("failed",), failed_root="validate_data"))
    assert out["triggers_met"] == [] and any("validate_data" in f for f in out["facts"])


def test_user_supplied_degradation_report_triggers_but_is_never_invented():
    assert verdict(evidence(degraded="AUC dropped 0.12 in last week's shadow traffic"))["verdict"] == "retraining_recommended"
    assert verdict(evidence())["triggers_met"] == []


def test_missing_evidence_yields_insufficient_evidence_not_guesses():
    ev = Evidence(errors={"k8s_app_version": "unavailable: cluster unreachable", "dvc_dataset_version": "error: no repo"})
    out = verdict(ev)
    assert out["verdict"] == "insufficient_evidence" and out["facts"] == []
    assert any("cluster unreachable" in u for u in out["uncertainties"])


def test_provenance_mismatch_is_reported_as_uncertainty():
    ev = evidence()
    ev.deployed_run["provenance"]["dvc.dataset_version"] = "v1"
    assert any("provenance mismatch" in u for u in verdict(ev)["uncertainties"])
