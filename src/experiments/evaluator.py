from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.environment.simulation import SimulationEnv
from src.experiments.metrics import collect_metrics, empty_metrics
from src.experiments.scenarios import ScenarioConfig, build_scenario
from src.strategies.heuristics import get_scheduler_by_id


@dataclass
class StrategyResult:
    strategy_id: str
    metrics: Dict[str, float]
    feasible: bool = True


@dataclass
class LabelResult:
    strategy_id: str
    scores: Dict[str, float]


def evaluate_strategy(config: ScenarioConfig, strategy_id: str, repeats: int = 1) -> StrategyResult:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    collected: List[Dict[str, float]] = []
    for repeat in range(repeats):
        scenario = build_scenario(
            ScenarioConfig(
                seed=config.seed + repeat,
                num_tasks=config.num_tasks,
                num_resources=config.num_resources,
                arrival_profile=config.arrival_profile,
                fault_profile=config.fault_profile,
            )
        )
        env = SimulationEnv(scheduler=get_scheduler_by_id(strategy_id))
        for resource in scenario.resources:
            env.add_resource(resource)
        for task in scenario.tasks:
            env.add_task(task)
        for event in scenario.events:
            env.inject_resource_fault(event["resource_id"], event["delay"], event["duration"])
        while env.event_queue:
            env.step()
        collected.append(collect_metrics(env))

    keys = empty_metrics().keys()
    averaged = {key: sum(row[key] for row in collected) / len(collected) for key in keys}
    return StrategyResult(strategy_id=strategy_id, metrics=averaged)


def evaluate_candidates(config: ScenarioConfig, strategy_ids: Iterable[str], repeats: int = 1) -> List[StrategyResult]:
    return [evaluate_strategy(config, strategy_id, repeats=repeats) for strategy_id in strategy_ids]


def select_demonstration_label(results: Iterable[StrategyResult], objective_weights: Dict[str, float]) -> LabelResult:
    results = list(results)
    if not results:
        raise ValueError("results must not be empty")
    time_values = [result.metrics["average_completion_time"] for result in results]
    low, high = min(time_values), max(time_values)
    span = high - low
    scored = []
    for result in results:
        normalized_time = (result.metrics["average_completion_time"] - low) / span if span else 0.0
        weighted_loss = objective_weights.get("time", 0.0) * normalized_time
        scored.append((weighted_loss, result.strategy_id, {"weighted_loss": weighted_loss, "normalized_time": normalized_time}))
    _, strategy_id, scores = min(scored, key=lambda item: (item[0], item[1]))
    return LabelResult(strategy_id=strategy_id, scores=scores)
