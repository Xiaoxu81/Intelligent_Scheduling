from src.experiments.demonstrations import generate_demonstrations
from src.experiments.scenarios import ScenarioConfig


def test_demonstrations_contain_report_state_and_strategy_label():
    rows = generate_demonstrations(
        [ScenarioConfig(seed=5, num_tasks=3, num_resources=2)],
        strategy_ids=["C01", "C03"],
        repeats=1,
    )

    assert len(rows) == 1
    state, action, metadata = rows[0]
    assert set(state) == {"system", "tasks", "resources", "weights"}
    assert action in {0, 2}
    assert metadata["strategy_id"] in {"C01", "C03"}
    assert metadata["seed"] == 5
