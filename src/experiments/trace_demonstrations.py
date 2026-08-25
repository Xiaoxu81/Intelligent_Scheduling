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
