from src.experiments.scenarios import ScenarioConfig, build_scenario
from src.experiments.metrics import empty_metrics
from src.experiments.io import write_experiment_result
from tempfile import TemporaryDirectory


def test_same_seed_builds_same_report_aligned_scenario():
    config = ScenarioConfig(seed=7, num_tasks=4, num_resources=2)
    first = build_scenario(config)
    second = build_scenario(config)

    assert first.to_dict() == second.to_dict()
    assert set(first.tasks[0].objective_weights) == {
        "time",
        "throughput",
        "cost",
        "stability",
    }


def test_empty_metrics_contains_report_core_metrics():
    metrics = empty_metrics()

    for key in (
        "average_completion_time",
        "deadline_satisfaction_rate",
        "throughput",
        "resource_utilization",
        "failure_rate",
        "starvation_risk",
        "recovery_time",
        "decision_time",
    ):
        assert key in metrics


def test_experiment_result_writes_traceable_json_and_csv():
    with TemporaryDirectory() as tmp_path:
        paths = write_experiment_result(
            tmp_path,
            metadata={"seed": 7, "method": "FCFS"},
            task_rows=[{"task_id": "T1", "completion_time": 3.0}],
            summary={"throughput": 1.0},
        )

        assert paths["json"].exists()
        assert paths["tasks_csv"].exists()
        assert '"seed": 7' in paths["json"].read_text(encoding="utf-8")
