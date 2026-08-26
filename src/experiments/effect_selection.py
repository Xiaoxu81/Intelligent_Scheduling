"""Feasibility, Pareto, and risk-aware selection over effect records."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

from src.experiments.effect_schema import EFFECT_KEYS, MAXIMIZE_KEYS, MINIMIZE_KEYS, OutcomeRecord


def filter_feasible(records: Iterable[OutcomeRecord]) -> List[OutcomeRecord]:
    return [record for record in records if record.feasible]


def _better_or_equal(left: float, right: float, key: str) -> bool:
    return left >= right if key in MAXIMIZE_KEYS else left <= right


def _strictly_better(left: float, right: float, key: str) -> bool:
    return left > right if key in MAXIMIZE_KEYS else left < right


def dominates(left: OutcomeRecord, right: OutcomeRecord) -> bool:
    """Return whether left is no worse on every effect and strictly better on one."""
    if not left.feasible:
        return False
    return all(_better_or_equal(left.metrics[key], right.metrics[key], key) for key in EFFECT_KEYS) and any(
        _strictly_better(left.metrics[key], right.metrics[key], key) for key in EFFECT_KEYS
    )


def pareto_front(records: Iterable[OutcomeRecord]) -> List[OutcomeRecord]:
    candidates = filter_feasible(records)
    return [candidate for candidate in candidates if not any(dominates(other, candidate) for other in candidates if other is not candidate)]


def _score(value: float, low: float, high: float, key: str) -> float:
    if high <= low:
        return 1.0
    fraction = (value - low) / (high - low)
    fraction = max(0.0, min(1.0, fraction))
    return fraction if key in MAXIMIZE_KEYS else 1.0 - fraction


def select_risk_constrained(
    records: Iterable[OutcomeRecord],
    demand: Mapping[str, object],
    risk_limits: Mapping[str, float],
) -> OutcomeRecord:
    """Select a Pareto candidate after hard risk filtering and demand preference."""
    feasible = filter_feasible(records)
    if not feasible:
        raise ValueError("no feasible strategy records")
    safe = [
        record for record in feasible
        if all(record.metrics.get(key, 0.0) <= limit for key, limit in risk_limits.items())
    ]
    candidates = pareto_front(safe) if safe else None
    if not candidates:
        # If every candidate violates a constraint, minimize normalized violation
        # first; preference is only used to break equal violations.
        def violation(record):
            return sum(max(0.0, record.metrics.get(key, 0.0) - limit) / max(limit, 1e-6) for key, limit in risk_limits.items())
        best_violation = min(violation(record) for record in feasible)
        candidates = [record for record in feasible if abs(violation(record) - best_violation) < 1e-12]
        candidates = pareto_front(candidates) or candidates

    bounds = {
        key: (min(record.metrics[key] for record in candidates), max(record.metrics[key] for record in candidates))
        for key in EFFECT_KEYS
    }
    weights = dict(demand.get("weights", {}))

    def preference_score(record: OutcomeRecord) -> float:
        scores = {key: _score(record.metrics[key], *bounds[key], key) for key in EFFECT_KEYS}
        stability_score = 0.5 * scores["failure_risk"] + 0.5 * scores["recovery_time"]
        return (
            weights.get("time", 0.25) * scores["completion_time"]
            + weights.get("throughput", 0.25) * scores["throughput"]
            + weights.get("cost", 0.25) * scores["resource_utilization"]
            + weights.get("stability", 0.25) * stability_score
        )

    return max(candidates, key=lambda record: (preference_score(record), record.strategy_id))
