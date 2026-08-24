from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.experiments.evaluator import evaluate_candidates, select_demonstration_label
from src.experiments.scenarios import ScenarioConfig, build_scenario


def scenario_to_observation(scenario, max_queue_size: int = 10, max_resource_size: int = 4) -> Dict[str, np.ndarray]:
    task_rows = np.zeros((max_queue_size, 9), dtype=np.float32) - 1.0
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
    return {"system": system, "tasks": task_rows, "resources": resource_rows, "weights": weights}


def generate_demonstrations(
    configs: Iterable[ScenarioConfig], strategy_ids: Iterable[str], repeats: int = 1
) -> List[Tuple[Dict[str, np.ndarray], int, Dict[str, object]]]:
    rows = []
    strategy_ids = list(strategy_ids)
    for config in configs:
        results = evaluate_candidates(config, strategy_ids, repeats=repeats)
        scenario = build_scenario(config)
        label = select_demonstration_label(results, scenario.tasks[0].objective_weights)
        rows.append(
            (
                scenario_to_observation(scenario),
                int(label.strategy_id[1:]) - 1,
                {
                    "seed": config.seed,
                    "strategy_id": label.strategy_id,
                    "scores": label.scores,
                },
            )
        )
    return rows
