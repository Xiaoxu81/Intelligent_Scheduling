import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace, load_v2018_rows
from src.experiments.metrics import collect_metrics
from src.models.drl_agent import PPOAgent


DEFAULT_STRATEGY_IDS = ["C01", "C03", "C04", "C05", "C09"]


def run_trace_training(
    trace: AlibabaTrace,
    seed: int = 0,
    episodes: int = 100,
    max_steps: int = 200,
    output_dir: Optional[Union[str, Path]] = None,
    k_epochs: int = 10,
    strategy_ids=None,
    expert_data=None,
    bc_epochs: int = 0,
    resume_from: Optional[Union[str, Path]] = None,
    max_assignments_per_step: Optional[int] = 1,
    fault_resource: Optional[str] = None,
    fault_at: Optional[float] = None,
    fault_duration: float = 0.0,
    fault_profiles=None,
) -> Dict[str, Any]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")

    torch.manual_seed(seed)
    traces = list(trace) if isinstance(trace, (list, tuple)) else [trace]
    if not traces or any(not isinstance(item, AlibabaTrace) for item in traces):
        raise ValueError("trace must be an AlibabaTrace or a non-empty sequence of AlibabaTrace objects")
    strategy_ids = list(strategy_ids or DEFAULT_STRATEGY_IDS)
    if fault_profiles is None:
        fault_profiles = [{"resource": fault_resource, "at": fault_at, "duration": fault_duration}]
    else:
        fault_profiles = [dict(profile) for profile in fault_profiles]
        if not fault_profiles:
            raise ValueError("fault_profiles must not be empty")
    initial_fault = fault_profiles[0]
    env = TraceSchedulingEnv(
        traces[0],
        max_queue_size=10,
        max_resource_size=4,
        strategy_ids=strategy_ids,
        max_assignments_per_step=max_assignments_per_step,
        fault_resource=initial_fault.get("resource"),
        fault_at=initial_fault.get("at"),
        fault_duration=initial_fault.get("duration", 0.0),
    )
    agent = PPOAgent(max_queue_size=10, num_strategies=len(strategy_ids), K_epochs=k_epochs)
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        agent.model.load_state_dict(state)
        agent.model_old.load_state_dict(state)
    if bc_epochs < 0:
        raise ValueError("bc_epochs must be non-negative")
    if bc_epochs and not expert_data:
        raise ValueError("expert_data is required when bc_epochs is positive")
    if bc_epochs:
        invalid_actions = [action for _, action, *_ in expert_data if not 0 <= int(action) < len(strategy_ids)]
        if invalid_actions:
            raise ValueError("expert_data actions must use the local strategy index")
        agent.pretrain_bc([(row[0], row[1]) for row in expert_data], epochs=bc_epochs)
    logs = []
    strategy_usage = [0] * agent.num_strategies

    for episode in range(episodes):
        env.trace = traces[episode % len(traces)]
        episode_fault = fault_profiles[episode % len(fault_profiles)]
        env.fault_resource = episode_fault.get("resource")
        env.fault_at = episode_fault.get("at")
        env.fault_duration = episode_fault.get("duration", 0.0)
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
        if not terminated and not truncated and len(memory) >= max_steps:
            truncated = True
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
        "data_source": traces[0].metadata.get("data_source", "trace"),
        "trace_metadata": [item.metadata for item in traces],
        "trace_windows": [item.metadata.get("window", item.metadata.get("time_window", index)) for index, item in enumerate(traces)],
        "simulation_engine": "src.environment.simulation.SimulationEnv",
        "seed": seed,
        "episodes": logs,
        "strategy_ids": strategy_ids,
        "behavior_cloning": {"enabled": bool(bc_epochs), "epochs": bc_epochs, "samples": len(expert_data or [])},
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "fault_profile": {
            "resource": initial_fault.get("resource"),
            "at": initial_fault.get("at"),
            "duration": initial_fault.get("duration") if initial_fault.get("resource") is not None else None,
        },
        "fault_profiles": fault_profiles,
        "strategy_usage": {strategy_id: count for strategy_id, count in zip(strategy_ids, strategy_usage)},
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "training_log.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        torch.save(agent.model.state_dict(), output_path / "ppo_model.pt")
    return result


def evaluate_trace_policy(
    trace: AlibabaTrace,
    model_path: Union[str, Path],
    max_steps: int = 200,
    seed: int = 0,
    strategy_ids=None,
    max_assignments_per_step: Optional[int] = 1,
    fault_resource: Optional[str] = None,
    fault_at: Optional[float] = None,
    fault_duration: float = 0.0,
) -> Dict[str, Any]:
    strategy_ids = list(strategy_ids or DEFAULT_STRATEGY_IDS)
    env = TraceSchedulingEnv(
        trace,
        max_queue_size=10,
        max_resource_size=4,
        strategy_ids=strategy_ids,
        max_assignments_per_step=max_assignments_per_step,
        fault_resource=fault_resource,
        fault_at=fault_at,
        fault_duration=fault_duration,
    )
    agent = PPOAgent(max_queue_size=10, num_strategies=len(strategy_ids), K_epochs=1)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    agent.model.load_state_dict(state)
    agent.model_old.load_state_dict(state)

    obs, _ = env.reset(seed=seed)
    usage = [0] * agent.num_strategies
    terminated = False
    truncated = False
    for _ in range(max_steps):
        action = int(agent.get_strategy_probs(obs).argmax())
        usage[action] += 1
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if not terminated and not truncated:
        truncated = True
    return {
        "data_source": trace.metadata.get("data_source", "trace"),
        "metrics": collect_metrics(env.sim),
        "steps": sum(usage),
        "terminated": terminated,
        "truncated": truncated,
        "strategy_ids": strategy_ids,
        "strategy_usage": {strategy_id: count for strategy_id, count in zip(strategy_ids, usage)},
    }


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
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGY_IDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/trace-ppo")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-assignments-per-step", type=int, default=1)
    parser.add_argument("--fault-resource")
    parser.add_argument("--fault-at", type=float)
    parser.add_argument("--fault-duration", type=float, default=0.0)
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
        strategy_ids=parsed.strategies,
        resume_from=parsed.resume_from,
        max_assignments_per_step=parsed.max_assignments_per_step,
        fault_resource=parsed.fault_resource,
        fault_at=parsed.fault_at,
        fault_duration=parsed.fault_duration,
    )
    print(json.dumps({"episodes": len(result["episodes"]), "output": parsed.output}, indent=2))


if __name__ == "__main__":
    main()
