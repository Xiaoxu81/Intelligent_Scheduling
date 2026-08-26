from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.experiments.evaluator import evaluate_candidates, select_demonstration_label
from src.experiments.scenarios import ScenarioConfig, build_scenario
from src.models.task_demand import DEMAND_KEYS


def scenario_to_observation(scenario, max_queue_size: int = 10, max_resource_size: int = 4) -> Dict[str, np.ndarray]:
    task_rows = np.zeros((max_queue_size, 9), dtype=np.float32) - 1.0
    demand_rows = np.zeros((max_queue_size, len(DEMAND_KEYS)), dtype=np.float32) - 1.0
    for i, task in enumerate(scenario.tasks[:max_queue_size]):
        task_rows[i] = [
            task.priority,
            task.duration,
            task.arrival_time,
            task.deadline if task.deadline is not None else -1.0,
            task.remaining_time,
            1.0 if task.is_realtime else 0.0,
            0.0,
            task.capability_requirements.get("machine", 0.0),
            float(len(task.dependencies)),
        ]
        demand = task.demand_features(
            current_time=0.0,
            feasible_resource_count=sum(
                1 for resource in scenario.resources
                if all(resource.capabilities.get(key, 0.0) >= value for key, value in task.capability_requirements.items())
            ),
            total_resource_count=len(scenario.resources),
        )
        demand_rows[i] = [demand[key] for key in DEMAND_KEYS]
    resource_rows = np.zeros((max_resource_size, 5), dtype=np.float32) - 1.0
    for i, resource in enumerate(scenario.resources[:max_resource_size]):
        resource_rows[i] = [
            resource.capacity,
            resource.capabilities.get("machine", 0.0),
            resource.reliability,
            0.0,
            0.0,
        ]
    first = scenario.tasks[0] if scenario.tasks else None
    weights = np.array(
        [first.objective_weights[key] for key in ("time", "throughput", "cost", "stability")]
        if first else [0.25] * 4,
        dtype=np.float32,
    )
    system = np.array([len(scenario.tasks), len(scenario.resources), 0, 0, 0, 0], dtype=np.float32)
    return {"system": system, "tasks": task_rows, "demands": demand_rows, "resources": resource_rows, "weights": weights}


def generate_demonstrations(
    configs: Iterable[ScenarioConfig],
    strategy_ids: Iterable[str],
    repeats: int = 1,
    action_strategy_ids: Iterable[str] = None,
) -> List[Tuple[Dict[str, np.ndarray], int, Dict[str, object]]]:
    rows = []
    strategy_ids = list(strategy_ids)
    action_strategy_ids = list(action_strategy_ids) if action_strategy_ids is not None else None
    if action_strategy_ids is not None and any(strategy_id not in action_strategy_ids for strategy_id in strategy_ids):
        raise ValueError("action_strategy_ids must contain every demonstrated strategy")
    for config in configs:
        results = evaluate_candidates(config, strategy_ids, repeats=repeats)
        scenario = build_scenario(config)
        label = select_demonstration_label(results, scenario.tasks[0].objective_weights)
        rows.append(
            (
                scenario_to_observation(scenario),
                (
                    action_strategy_ids.index(label.strategy_id)
                    if action_strategy_ids is not None
                    else int(label.strategy_id[1:]) - 1
                ),
                {
                    "seed": config.seed,
                    "strategy_id": label.strategy_id,
                    "scores": label.scores,
                },
            )
        )
    return rows
