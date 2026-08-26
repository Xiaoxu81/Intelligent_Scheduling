"""Shared schema and direction-aware normalization for strategy effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


EFFECT_KEYS = (
    "completion_time",
    "throughput",
    "resource_utilization",
    "failure_risk",
    "deadline_risk",
    "recovery_time",
)

MINIMIZE_KEYS = {"completion_time", "failure_risk", "deadline_risk", "recovery_time"}
MAXIMIZE_KEYS = {"throughput", "resource_utilization"}


@dataclass
class OutcomeRecord:
    strategy_id: str
    feasible: bool
    metrics: Dict[str, float]
    next_state: Dict[str, float]
    metadata: Dict[str, object]

    def __post_init__(self) -> None:
        missing = [key for key in EFFECT_KEYS if key not in self.metrics]
        if missing:
            raise ValueError(f"metrics missing effect keys: {missing}")
        self.metrics = {key: float(self.metrics[key]) for key in EFFECT_KEYS}


def normalize_outcome(
    metrics: Mapping[str, float], bounds: Mapping[str, Tuple[float, float]]
) -> Dict[str, float]:
    """Convert raw metrics to a higher-is-better score in ``[0, 1]``."""
    scores: Dict[str, float] = {}
    for key in EFFECT_KEYS:
        if key not in metrics or key not in bounds:
            raise ValueError(f"missing metric bounds for {key}")
        low, high = bounds[key]
        if high <= low:
            raise ValueError(f"invalid bounds for {key}: {(low, high)}")
        value = float(metrics[key])
        fraction = max(0.0, min(1.0, (value - low) / (high - low)))
        if key in MINIMIZE_KEYS:
            fraction = 1.0 - fraction
        elif key not in MAXIMIZE_KEYS:
            raise ValueError(f"unknown effect direction for {key}")
        scores[key] = fraction
    return scores
