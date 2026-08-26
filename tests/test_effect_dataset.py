from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace
from src.experiments.effect_dataset import generate_counterfactual_records
from src.models.resource import Resource
from src.models.task import Task


def test_counterfactual_records_share_initial_state_and_respect_horizon():
    trace = AlibabaTrace(
        tasks=[
            Task("j1:M1", priority=3, duration=2.0, arrival_time=0.0, deadline=6.0),
            Task("j1:M2", priority=6, duration=1.0, arrival_time=0.0, deadline=5.0),
        ],
        resources=[Resource("m1", "Machine", capacity=1.0, capabilities={"machine": 2.0})],
        metadata={"data_source": "synthetic_test", "window": "small"},
    )

    def env_factory():
        return TraceSchedulingEnv(trace, strategy_ids=["C01", "C04"], max_assignments_per_step=1)

    records = generate_counterfactual_records(env_factory, ["C01", "C04"], horizon=2, seed=7)

    assert len(records) == 2
    assert {record.strategy_id for record in records} == {"C01", "C04"}
    assert len({record.metadata["initial_state_fingerprint"] for record in records}) == 1
    assert records[0].initial_state["system"] == records[1].initial_state["system"]
    assert all(record.metadata["steps"] <= 2 for record in records)
    assert all("completion_time" in record.metrics for record in records)
