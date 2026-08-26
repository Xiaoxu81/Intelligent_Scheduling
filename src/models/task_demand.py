"""Derive interpretable, continuous task-demand features from observable state."""

from __future__ import annotations

from typing import Dict, Optional


DEMAND_KEYS = (
    "urgency",
    "criticality",
    "throughput_preference",
    "cost_sensitivity",
    "stability_requirement",
    "resource_scarcity",
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def derive_task_demand(
    *,
    priority: int,
    duration: float,
    arrival_time: float,
    current_time: float,
    deadline: Optional[float],
    dependency_count: int,
    downstream_count: int = 0,
    failure_penalty: float = 0.0,
    feasible_resource_count: int = 0,
    total_resource_count: int = 0,
    cost_sensitivity: float = 0.5,
    stability_requirement: float = 0.5,
) -> Dict[str, object]:
    """Compute task demand scores and normalized objective preferences.

    The scores are continuous and derived from task metadata plus the current
    system state. They are deliberately separate from the later strategy
    effect model: demand describes what the task needs, not which strategy wins.
    """
    priority_score = _clip(priority / 10.0)
    horizon = max(float(duration) * 2.0, 1.0)
    waiting_time = max(float(current_time) - float(arrival_time), 0.0)
    wait_pressure = _clip(waiting_time / horizon)
    if deadline is None:
        deadline_pressure = 0.0
    else:
        slack = float(deadline) - float(current_time) - max(float(duration), 0.0)
        deadline_pressure = _clip(1.0 - slack / horizon)
    urgency = _clip(0.6 * deadline_pressure + 0.2 * wait_pressure + 0.2 * priority_score)

    downstream_impact = _clip(downstream_count / 4.0)
    dependency_impact = _clip(dependency_count / 3.0)
    total_resources = max(int(total_resource_count), 0)
    feasible_resources = max(int(feasible_resource_count), 0)
    if total_resources == 0:
        resource_scarcity = 1.0
    else:
        resource_scarcity = _clip(1.0 - feasible_resources / total_resources)
    criticality = _clip(
        0.30 * priority_score
        + 0.20 * dependency_impact
        + 0.25 * downstream_impact
        + 0.15 * _clip(failure_penalty)
        + 0.10 * resource_scarcity
    )

    throughput = _clip(0.55 * (1.0 - urgency) + 0.45 * (1.0 - criticality))
    cost = _clip(cost_sensitivity)
    stability = _clip(max(stability_requirement, 0.6 * criticality + 0.4 * resource_scarcity))
    raw_weights = {
        "time": 0.15 + 0.55 * urgency,
        "throughput": 0.15 + 0.40 * throughput,
        "cost": 0.15 + 0.35 * cost,
        "stability": 0.15 + 0.45 * stability,
    }
    total = sum(raw_weights.values()) or 1.0
    weights = {key: value / total for key, value in raw_weights.items()}
    scores = {
        "urgency": urgency,
        "criticality": criticality,
        "throughput_preference": throughput,
        "cost_sensitivity": cost,
        "stability_requirement": stability,
        "resource_scarcity": resource_scarcity,
    }
    return {**scores, "weights": weights}
