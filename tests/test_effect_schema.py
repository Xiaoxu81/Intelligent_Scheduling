from src.experiments.effect_schema import EFFECT_KEYS, OutcomeRecord, normalize_outcome


def test_normalize_outcome_uses_metric_direction_consistently():
    metrics = {
        "completion_time": 5.0,
        "throughput": 5.0,
        "resource_utilization": 0.75,
        "failure_risk": 0.10,
        "deadline_risk": 0.20,
        "recovery_time": 2.0,
    }
    bounds = {
        "completion_time": (0.0, 10.0),
        "throughput": (0.0, 10.0),
        "resource_utilization": (0.0, 1.0),
        "failure_risk": (0.0, 1.0),
        "deadline_risk": (0.0, 1.0),
        "recovery_time": (0.0, 10.0),
    }
    scores = normalize_outcome(metrics, bounds)
    assert scores["completion_time"] == 0.5
    assert scores["throughput"] == 0.5
    assert scores["resource_utilization"] == 0.75
    assert scores["failure_risk"] == 0.9
    assert scores["deadline_risk"] == 0.8
    assert scores["recovery_time"] == 0.8


def test_outcome_record_requires_all_effect_keys():
    record = OutcomeRecord(
        strategy_id="C01",
        feasible=True,
        metrics={key: 0.0 for key in EFFECT_KEYS},
        next_state={"pending_workload": 1.0},
        metadata={"seed": 1},
    )
    assert record.strategy_id == "C01"
    assert tuple(record.metrics) == EFFECT_KEYS
