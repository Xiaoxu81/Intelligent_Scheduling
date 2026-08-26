from src.models.task import Task
from src.models.task_demand import DEMAND_KEYS, derive_task_demand


def test_urgency_increases_when_deadline_slack_shrinks():
    relaxed = derive_task_demand(
        priority=5, duration=4, arrival_time=0, current_time=0, deadline=20,
        dependency_count=0, downstream_count=0, feasible_resource_count=2, total_resource_count=2,
    )
    urgent = derive_task_demand(
        priority=5, duration=4, arrival_time=0, current_time=15, deadline=20,
        dependency_count=0, downstream_count=0, feasible_resource_count=2, total_resource_count=2,
    )
    assert urgent["urgency"] > relaxed["urgency"]
    assert urgent["weights"]["time"] > relaxed["weights"]["time"]


def test_criticality_increases_with_downstream_impact_and_resource_scarcity():
    ordinary = derive_task_demand(
        priority=3, duration=4, arrival_time=0, current_time=0, deadline=20,
        dependency_count=0, downstream_count=0, feasible_resource_count=3, total_resource_count=3,
    )
    critical = derive_task_demand(
        priority=9, duration=4, arrival_time=0, current_time=0, deadline=20,
        dependency_count=2, downstream_count=4, failure_penalty=1.0,
        feasible_resource_count=1, total_resource_count=3,
    )
    assert critical["criticality"] > ordinary["criticality"]
    assert critical["resource_scarcity"] > ordinary["resource_scarcity"]
    assert critical["weights"]["stability"] > ordinary["weights"]["stability"]


def test_task_exposes_demand_features_and_normalized_weights():
    task = Task("T1", priority=8, duration=5, arrival_time=0, deadline=10, dependencies=["T0"])
    demand = task.demand_features(current_time=4, downstream_count=2, feasible_resource_count=1, total_resource_count=2)
    assert all(key in demand for key in DEMAND_KEYS)
    assert all(0.0 <= demand[key] <= 1.0 for key in DEMAND_KEYS)
    assert abs(sum(task.objective_weights.values()) - 1.0) < 1e-6
