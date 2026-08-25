import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.environment.gym_wrapper import SchedulingEnv
from src.models.drl_agent import PPOAgent


def run_training(
    seed: int = 0,
    episodes: int = 100,
    max_steps: int = 200,
    update_every: int = 1,
    output_dir: Optional[Union[str, Path]] = None,
    k_epochs: int = 10,
) -> Dict[str, Any]:
    if episodes < 1 or max_steps < 1 or update_every < 1:
        raise ValueError("episodes, max_steps and update_every must be positive")

    torch.manual_seed(seed)
    env = SchedulingEnv(max_queue_size=10, num_strategies=12)
    agent = PPOAgent(max_queue_size=10, K_epochs=k_epochs)
    logs = []
    memory = []
    strategy_usage = [0] * agent.num_strategies

    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + episode - 1)
        episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False
        while not terminated and not truncated and episode_steps < max_steps:
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
            episode_reward += float(reward)
            episode_steps += 1
            obs = next_obs

        if episode % update_every == 0 and memory:
            agent.update(memory)
            memory = []
        logs.append({
            "episode": episode,
            "episode_reward": episode_reward,
            "steps": episode_steps,
            "terminated": terminated,
            "truncated": truncated,
        })

    result = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a report-aligned PPO training experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--k-epochs", type=int, default=10)
    parser.add_argument("--output", default="results/ppo")
    args = parser.parse_args()
    result = run_training(
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        update_every=args.update_every,
        output_dir=args.output,
        k_epochs=args.k_epochs,
    )
    print(json.dumps({"episodes": len(result["episodes"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
