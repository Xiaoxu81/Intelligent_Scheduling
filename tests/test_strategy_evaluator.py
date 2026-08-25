from src.experiments.evaluator import evaluate_candidates, select_demonstration_label
from src.experiments.scenarios import ScenarioConfig


def test_candidate_evaluation_returns_all_requested_methods_and_metrics():
    results = evaluate_candidates(
        ScenarioConfig(seed=3, num_tasks=3, num_resources=1),
        strategy_ids=["C01", "C03"],
        repeats=1,
    )

    assert [result.strategy_id for result in results] == ["C01", "C03"]
    assert all("average_completion_time" in result.metrics for result in results)
    assert all(result.metrics["resource_utilization"] > 0.0 for result in results)


def test_demonstration_label_uses_objective_weights_and_stable_tie_breaking():
    results = evaluate_candidates(
        ScenarioConfig(seed=3, num_tasks=3, num_resources=1),
        strategy_ids=["C03", "C01"],
        repeats=1,
    )
    label = select_demonstration_label(results, {"time": 1.0, "throughput": 0.0, "cost": 0.0, "stability": 0.0})

    assert label.strategy_id in {"C01", "C03"}
    assert "weighted_loss" in label.scores


def test_fault_scenario_records_recovery_and_decision_metrics():
    result = evaluate_candidates(
        ScenarioConfig(seed=3, num_tasks=3, num_resources=1, fault_profile="single"),
        strategy_ids=["C01"],
        repeats=1,
    )[0]

    assert result.metrics["recovery_time"] > 0.0
    assert result.metrics["decision_time"] >= 0.0
