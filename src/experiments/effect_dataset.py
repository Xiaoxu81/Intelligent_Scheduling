"""Generate counterfactual multi-step outcomes for candidate strategies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np

from src.experiments.effect_schema import EFFECT_KEYS, OutcomeRecord
from src.experiments.metrics import collect_metrics
from src.models.task import TaskStatus


def _fingerprint(observation) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(observation[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _is_feasible(env) -> bool:
    resources = list(env.sim.resources.values())
    for task in env.sim.tasks.values():
        if task.status != TaskStatus.READY:
            continue
        compatible = [
            resource
            for resource in resources
            if all(resource.capabilities.get(key, 0.0) >= value for key, value in task.capability_requirements.items())
        ]
        if not compatible:
            return False
    return True


def _next_state(env):
    tasks = list(env.sim.tasks.values())
    resources = list(env.sim.resources.values())
    return {
        "current_time": float(env.sim.current_time),
        "ready_tasks": float(sum(task.status == TaskStatus.READY for task in tasks)),
        "pending_tasks": float(sum(task.status == TaskStatus.PENDING for task in tasks)),
        "failed_tasks": float(sum(task.status == TaskStatus.FAILED for task in tasks)),
        "idle_resources": float(sum(resource.status.value == "IDLE" for resource in resources)),
        "pending_workload": float(sum(task.remaining_time for task in tasks if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED))),
    }


def generate_counterfactual_records(
    env_factory: Callable[[], object],
    strategy_ids: Iterable[str],
    horizon: int = 3,
    seed: int = 0,
) -> List[OutcomeRecord]:
    """Run each candidate from the same initial state for at most ``horizon`` steps."""
    strategy_ids = list(strategy_ids)
    if not strategy_ids:
        raise ValueError("strategy_ids must not be empty")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    records = []
    for action, strategy_id in enumerate(strategy_ids):
        env = env_factory()
        observation, _ = env.reset(seed=seed)
        initial_fingerprint = _fingerprint(observation)
        feasible = _is_feasible(env)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        for _ in range(horizon):
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        raw = collect_metrics(env.sim)
        metrics = {
            "completion_time": raw["average_completion_time"],
            "throughput": raw["throughput"],
            "resource_utilization": raw["resource_utilization"],
            "failure_risk": raw["failure_rate"],
            "deadline_risk": 1.0 - raw["deadline_satisfaction_rate"],
            "recovery_time": raw["recovery_time"],
        }
        metadata = {
            "data_source": getattr(getattr(env, "trace", None), "metadata", {}).get("data_source", "unknown"),
            "window": getattr(getattr(env, "trace", None), "metadata", {}).get("window", "unknown"),
            "seed": seed,
            "horizon": horizon,
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "initial_state_fingerprint": initial_fingerprint,
        }
        records.append(OutcomeRecord(strategy_id, feasible, metrics, _next_state(env), metadata))
    return records


def save_effect_dataset(records: Iterable[OutcomeRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "effect_keys": list(EFFECT_KEYS),
        "records": [
            {
                "strategy_id": record.strategy_id,
                "feasible": record.feasible,
                "metrics": record.metrics,
                "next_state": record.next_state,
                "metadata": record.metadata,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_effect_dataset(path: str | Path) -> List[OutcomeRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("effect_keys") != list(EFFECT_KEYS):
        raise ValueError("unsupported effect dataset schema")
    return [OutcomeRecord(**row) for row in payload["records"]]
