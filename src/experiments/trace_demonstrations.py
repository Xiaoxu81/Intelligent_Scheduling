from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace
from src.experiments.metrics import collect_metrics


def _run_fixed_policy(trace: AlibabaTrace, strategy_ids: List[str], action: int, max_steps: int):
    env = TraceSchedulingEnv(trace, max_queue_size=10, max_resource_size=4, strategy_ids=strategy_ids)
    observation, _ = env.reset(seed=0)
    states = []
    for _ in range(max_steps):
        states.append({key: np.array(value, copy=True) for key, value in observation.items()})
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env, states


def generate_trace_demonstrations(
    traces: Iterable[AlibabaTrace], strategy_ids: Iterable[str], max_steps: int = 500
) -> List[Tuple[Dict[str, np.ndarray], int, Dict[str, object]]]:
    """Create state/action samples from the best fixed candidate in each trace window."""
    traces = list(traces)
    strategy_ids = list(strategy_ids)
    if not traces or not strategy_ids:
        raise ValueError("traces and strategy_ids must not be empty")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    rows = []
    for trace in traces:
        candidates = []
        for action, strategy_id in enumerate(strategy_ids):
            env, _ = _run_fixed_policy(trace, strategy_ids, action, max_steps)
            metrics = collect_metrics(env.sim)
            candidates.append((metrics["average_completion_time"], strategy_id, action))
        _, expert_strategy, expert_action = min(candidates, key=lambda item: (item[0], item[1]))
        _, states = _run_fixed_policy(trace, strategy_ids, expert_action, max_steps)
        for state in states:
            rows.append(
                (
                    state,
                    expert_action,
                    {
                        "window": trace.metadata.get("window", trace.metadata.get("time_window", "unknown")),
                        "expert_strategy": expert_strategy,
                    },
                )
            )
    return rows


def _replay_history(trace, strategy_ids, history):
    env = TraceSchedulingEnv(
        trace,
        max_queue_size=10,
        max_resource_size=4,
        strategy_ids=strategy_ids,
        max_assignments_per_step=1,
    )
    observation, _ = env.reset(seed=0)
    for action in history:
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env, observation


def generate_local_trace_demonstrations(
    traces: Iterable[AlibabaTrace], strategy_ids: Iterable[str], max_steps: int = 500
) -> List[Tuple[Dict[str, np.ndarray], int, Dict[str, object]]]:
    """Label each fine-grained decision with the best immediate candidate action."""
    traces = list(traces)
    strategy_ids = list(strategy_ids)
    if not traces or not strategy_ids:
        raise ValueError("traces and strategy_ids must not be empty")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    rows = []
    for trace in traces:
        history = []
        for _ in range(max_steps):
            current_env, observation = _replay_history(trace, strategy_ids, history)
            if all(task.status.value in {"completed", "failed"} for task in current_env.sim.tasks.values()):
                break
            scored = []
            for action in range(len(strategy_ids)):
                candidate_env, _ = _replay_history(trace, strategy_ids, history)
                _, reward, terminated, truncated, _ = candidate_env.step(action)
                scored.append((float(reward), -action, terminated, truncated))
            _, negative_action, _, _ = max(scored)
            action = -negative_action
            rows.append(
                (
                    {key: np.array(value, copy=True) for key, value in observation.items()},
                    action,
                    {
                        "window": trace.metadata.get("window", trace.metadata.get("time_window", "unknown")),
                        "label_type": "local_greedy",
                        "strategy_id": strategy_ids[action],
                    },
                )
            )
            history.append(action)
            if scored[action][2] or scored[action][3]:
                break
    return rows


def _lookahead_score(trace, strategy_ids, history, first_action, horizon):
    future_history = list(history)
    total_reward = 0.0
    for depth in range(horizon):
        if depth == 0:
            action = first_action
        else:
            candidates = []
            for candidate in range(len(strategy_ids)):
                candidate_env, _ = _replay_history(trace, strategy_ids, future_history)
                _, reward, terminated, truncated, _ = candidate_env.step(candidate)
                candidates.append((float(reward), -candidate, terminated, truncated))
            _, negative_action, _, _ = max(candidates)
            action = -negative_action
        env, _ = _replay_history(trace, strategy_ids, future_history)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        future_history.append(action)
        if terminated or truncated:
            break
    return total_reward


def generate_lookahead_trace_demonstrations(
    traces: Iterable[AlibabaTrace],
    strategy_ids: Iterable[str],
    horizon: int = 3,
    max_steps: int = 500,
) -> List[Tuple[Dict[str, np.ndarray], int, Dict[str, object]]]:
    """Generate fine-grained labels using a short greedy rollout for each action."""
    traces = list(traces)
    strategy_ids = list(strategy_ids)
    if not traces or not strategy_ids:
        raise ValueError("traces and strategy_ids must not be empty")
    if horizon < 1 or max_steps < 1:
        raise ValueError("horizon and max_steps must be positive")

    rows = []
    for trace in traces:
        history = []
        for _ in range(max_steps):
            current_env, observation = _replay_history(trace, strategy_ids, history)
            if all(task.status.value in {"completed", "failed"} for task in current_env.sim.tasks.values()):
                break
            scored = [
                (_lookahead_score(trace, strategy_ids, history, action, horizon), -action)
                for action in range(len(strategy_ids))
            ]
            _, negative_action = max(scored)
            action = -negative_action
            rows.append(
                (
                    {key: np.array(value, copy=True) for key, value in observation.items()},
                    action,
                    {
                        "window": trace.metadata.get("window", trace.metadata.get("time_window", "unknown")),
                        "label_type": "lookahead_greedy",
                        "horizon": horizon,
                        "strategy_id": strategy_ids[action],
                    },
                )
            )
            history.append(action)
    return rows
