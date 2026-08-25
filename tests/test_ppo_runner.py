import json

from src.experiments.run_ppo import run_training


def test_ppo_runner_writes_report_ready_training_log(tmp_path):
    result = run_training(
        seed=4,
        episodes=2,
        max_steps=20,
        update_every=1,
        output_dir=tmp_path,
        k_epochs=1,
    )

    assert len(result["episodes"]) == 2
    assert all("episode_reward" in row for row in result["episodes"])
    log_path = tmp_path / "training_log.json"
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding="utf-8"))["seed"] == 4
