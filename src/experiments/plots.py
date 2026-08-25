import json
from pathlib import Path
from typing import Mapping, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_strategy_metric(summary: Mapping[str, Mapping[str, Mapping[str, float]]], metric: str, output_path: Union[str, Path]) -> Path:
    output = Path(output_path)
    methods = [method for method in sorted(summary) if metric in summary[method]]
    means = [summary[method][metric]["mean"] for method in methods]
    errors = [summary[method][metric]["std"] for method in methods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(methods, means, yerr=errors, capsize=4, color="#4C78A8")
    ax.set_xlabel("Strategy")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Strategy comparison: {metric}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_training_rewards(log_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    x = [row["episode"] for row in episodes]
    y = [row["episode_reward"] for row in episodes]
    output = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color="#F58518", linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO training reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
