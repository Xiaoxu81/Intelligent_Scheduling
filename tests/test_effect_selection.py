from src.experiments.effect_schema import EFFECT_KEYS, OutcomeRecord
from src.experiments.effect_selection import filter_feasible, pareto_front, select_risk_constrained


def record(strategy, completion, throughput, failure, recovery, feasible=True):
    return OutcomeRecord(
        strategy_id=strategy,
        feasible=feasible,
        metrics={
            "completion_time": completion,
            "throughput": throughput,
            "resource_utilization": 0.5,
            "failure_risk": failure,
            "deadline_risk": failure,
            "recovery_time": recovery,
        },
        next_state={"pending_workload": 1.0},
        metadata={},
    )


def test_selection_removes_infeasible_and_pareto_dominated_records():
    dominated = record("C01", completion=10, throughput=5, failure=0.2, recovery=5)
    winner = record("C04", completion=8, throughput=6, failure=0.1, recovery=3)
    infeasible = record("C05", completion=1, throughput=10, failure=0.0, recovery=0, feasible=False)
    assert filter_feasible([dominated, winner, infeasible]) == [dominated, winner]
    assert [row.strategy_id for row in pareto_front([dominated, winner, infeasible])] == ["C04"]


def test_risk_limit_overrides_shorter_completion_time():
    fast_risky = record("C01", completion=5, throughput=8, failure=0.3, recovery=1)
    slower_safe = record("C04", completion=8, throughput=7, failure=0.02, recovery=2)
    demand = {"weights": {"time": 0.8, "throughput": 0.1, "cost": 0.05, "stability": 0.05}}
    selected = select_risk_constrained([fast_risky, slower_safe], demand, {"failure_risk": 0.05, "deadline_risk": 0.05})
    assert selected.strategy_id == "C04"
