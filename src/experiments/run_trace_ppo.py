import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace, load_v2018_rows
from src.models.drl_agent import PPOAgent


def run_trace_training(
    trace: AlibabaTrace,
    seed: int = 0,
    episodes: int = 100,
    max_steps: int = 200,
    output_dir: Optional[Union[str, Path]] = None,
    k_epochs: int = 10,
) -> Dict[str, Any]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")

    torch.manual_seed(seed)
    env = TraceSchedulingEnv(trace, max_queue_size=10, max_resource_size=4)
    agent = PPOAgent(max_queue_size=10, K_epochs=k_epochs)
    logs = []
    strategy_usage = [0] * agent.num_strategies

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        memory = []
        reward_total = 0.0
        terminated = False
        truncated = False
        for step in range(max_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            memory.append({
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "value": value,
                "done": terminated or truncated,
            })
            strategy_usage[action] += 1
            reward_total += float(reward)
            obs = next_obs
            if terminated or truncated:
                break
        if memory:
            agent.update(memory)
        logs.append({
            "episode": episode + 1,
            "episode_reward": reward_total,
            "steps": len(memory),
            "terminated": terminated,
            "truncated": truncated,
        })

    result = {
        "data_source": trace.metadata.get("data_source", "trace"),
        "trace_metadata": trace.metadata,
        "simulation_engine": "src.environment.simulation.SimulationEnv",
        "seed": seed,
        "episodes": logs,
        "strategy_usage": {f"C{i + 1:02d}": count for i, count in enumerate(strategy_usage)},
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "training_log.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        torch.save(agent.model.state_dict(), output_path / "ppo_model.pt")
    return result


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on an Alibaba trace workload.")
    parser.add_argument("--machine-meta", required=True)
    parser.add_argument("--batch-task", required=True)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--limit-jobs", type=int, default=20)
    parser.add_argument("--limit-resources", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--k-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/trace-ppo")
    parsed = parser.parse_args(args)
    trace = load_v2018_rows(
        parsed.machine_meta,
        parsed.batch_task,
        start_time=parsed.start_time,
        end_time=parsed.end_time,
        limit_jobs=parsed.limit_jobs,
        limit_resources=parsed.limit_resources,
    )
    result = run_trace_training(
        trace,
        seed=parsed.seed,
        episodes=parsed.episodes,
        max_steps=parsed.max_steps,
        output_dir=parsed.output,
        k_epochs=parsed.k_epochs,
    )
    print(json.dumps({"episodes": len(result["episodes"]), "output": parsed.output}, indent=2))


if __name__ == "__main__":
    main()
