import json

from src.experiments.plots import plot_strategy_metric, plot_training_rewards


def test_report_plots_are_written_from_saved_data(tmp_path):
    summary = {
        "C01": {"throughput": {"mean": 1.0, "std": 0.1, "n": 2}},
        "C03": {"throughput": {"mean": 2.0, "std": 0.2, "n": 2}},
    }
    metric_path = plot_strategy_metric(summary, "throughput", tmp_path / "throughput.png")
    log_path = tmp_path / "training.json"
    log_path.write_text(json.dumps({"episodes": [{"episode": 1, "episode_reward": 2.0}]}), encoding="utf-8")
    reward_path = plot_training_rewards(log_path, tmp_path / "rewards.png")

    assert metric_path.exists()
    assert reward_path.exists()
