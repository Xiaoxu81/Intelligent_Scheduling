import numpy as np

from src.experiments.alibaba_trace import AlibabaTrace
from src.experiments.run_trace_ppo import evaluate_trace_policy, run_trace_training
from src.models.resource import Resource
from src.models.task import Task


def test_trace_ppo_runner_records_real_trace_provenance(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(trace, episodes=2, max_steps=3, k_epochs=1, output_dir=tmp_path)

    assert result["data_source"] == "alibaba_cluster_trace_v2018"
    assert len(result["episodes"]) == 2
    assert (tmp_path / "training_log.json").exists()


def test_trace_ppo_marks_step_limit_as_truncated():
    trace = AlibabaTrace(
        tasks=[
            Task("j1:M1", priority=1, duration=10.0, arrival_time=0.0),
            Task("j1:M2", priority=1, duration=10.0, arrival_time=0.0),
        ],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(trace, episodes=1, max_steps=1, k_epochs=1)

    assert result["episodes"][0]["truncated"] is True


def test_trace_ppo_evaluation_returns_comparable_metrics(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    run_trace_training(trace, episodes=1, max_steps=3, k_epochs=1, output_dir=tmp_path)

    evaluation = evaluate_trace_policy(trace, tmp_path / "ppo_model.pt", max_steps=3)

    assert "average_completion_time" in evaluation["metrics"]
    assert evaluation["data_source"] == "alibaba_cluster_trace_v2018"


def test_trace_ppo_evaluation_marks_partial_rollout(tmp_path):
    trace = AlibabaTrace(
        tasks=[
            Task("j1:M1", priority=1, duration=10.0, arrival_time=0.0),
            Task("j1:M2", priority=1, duration=10.0, arrival_time=0.0),
        ],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    run_trace_training(trace, episodes=1, max_steps=2, k_epochs=1, output_dir=tmp_path)

    evaluation = evaluate_trace_policy(trace, tmp_path / "ppo_model.pt", max_steps=1)

    assert evaluation["truncated"] is True


def test_trace_ppo_can_train_with_restricted_candidates(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(
        trace,
        episodes=1,
        max_steps=3,
        k_epochs=1,
        strategy_ids=["C04", "C09"],
        output_dir=tmp_path,
    )

    assert set(result["strategy_usage"]) == {"C04", "C09"}


def test_trace_ppo_can_train_on_multiple_trace_windows(tmp_path):
    traces = [
        AlibabaTrace(
            tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
            resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
            metadata={"data_source": "alibaba_cluster_trace_v2018", "window": "a"},
        ),
        AlibabaTrace(
            tasks=[Task("j2:M1", priority=1, duration=1.0, arrival_time=0.0)],
            resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
            metadata={"data_source": "alibaba_cluster_trace_v2018", "window": "b"},
        ),
    ]

    result = run_trace_training(traces, episodes=2, max_steps=3, k_epochs=1, output_dir=tmp_path)

    assert result["trace_windows"] == ["a", "b"]


def test_trace_ppo_can_resume_from_saved_model(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    first_dir = tmp_path / "first"
    run_trace_training(trace, episodes=1, max_steps=3, k_epochs=1, output_dir=first_dir)

    resumed = run_trace_training(
        trace,
        episodes=1,
        max_steps=3,
        k_epochs=1,
        resume_from=first_dir / "ppo_model.pt",
        output_dir=tmp_path / "resumed",
    )

    assert resumed["resumed_from"] == str(first_dir / "ppo_model.pt")


def test_trace_ppo_accepts_controlled_fault_configuration(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=4.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(
        trace,
        episodes=1,
        max_steps=10,
        k_epochs=1,
        fault_resource="m1",
        fault_at=1.0,
        fault_duration=2.0,
        output_dir=tmp_path,
    )

    assert result["fault_profile"]["resource"] == "m1"


def test_trace_ppo_can_cycle_normal_and_fault_profiles(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(
        trace,
        episodes=2,
        max_steps=3,
        k_epochs=1,
        fault_profiles=[
            {"resource": None, "at": None, "duration": 0.0},
            {"resource": "m1", "at": 0.5, "duration": 1.0},
        ],
        output_dir=tmp_path,
    )

    assert len(result["fault_profiles"]) == 2


def test_trace_ppo_can_keep_expert_regularization_during_updates(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    expert_data = [
        (
            {
                "tasks": np.zeros((10, 9), dtype=np.float32),
                "system": np.zeros(6, dtype=np.float32),
                "resources": np.zeros((4, 5), dtype=np.float32),
                "weights": np.full(4, 0.25, dtype=np.float32),
            },
            0,
            {},
        )
    ]

    result = run_trace_training(
        trace,
        episodes=1,
        max_steps=3,
        k_epochs=1,
        expert_data=expert_data,
        bc_epochs=1,
        bc_epochs_per_update=1,
        output_dir=tmp_path,
    )

    assert result["behavior_cloning"]["epochs_per_update"] == 1
