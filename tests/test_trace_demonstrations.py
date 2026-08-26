from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace
from src.experiments.trace_demonstrations import (
    generate_lookahead_trace_demonstrations,
    generate_local_trace_demonstrations,
    generate_trace_demonstrations,
)
from src.models.resource import Resource
from src.models.task import Task


def test_trace_demonstrations_use_local_candidate_actions():
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018", "window": "small"},
    )

    rows = generate_trace_demonstrations([trace], ["C01", "C04"], max_steps=3)

    assert rows
    state, action, metadata = rows[0]
    assert set(state) == {"system", "tasks", "demands", "resources", "weights", "global"}
    assert action in {0, 1}
    assert metadata["window"] == "small"
    assert metadata["expert_strategy"] in {"C01", "C04"}


def test_local_trace_demonstrations_label_each_decision():
    trace = AlibabaTrace(
        tasks=[
            Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0),
            Task("j1:M2", priority=5, duration=1.0, arrival_time=0.0),
        ],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018", "window": "local"},
    )

    rows = generate_local_trace_demonstrations([trace], ["C01", "C04"], max_steps=5)

    assert rows
    assert all(row[1] in {0, 1} for row in rows)
    assert all(row[2]["label_type"] == "local_greedy" for row in rows)


def test_lookahead_trace_demonstrations_record_horizon():
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018", "window": "lookahead"},
    )

    rows = generate_lookahead_trace_demonstrations([trace], ["C01", "C04"], horizon=3, max_steps=3)

    assert rows
    assert rows[0][2]["label_type"] == "lookahead_greedy"
    assert rows[0][2]["horizon"] == 3
