from src.models.task import Task
from src.models.resource import Resource


def test_task_has_report_aligned_objective_weights_and_capability_requirements():
    task = Task(
        "T1",
        priority=2,
        duration=3.0,
        capability_requirements={"machine": 2},
        objective_weights={"time": 0.7, "throughput": 0.1, "cost": 0.1, "stability": 0.1},
    )

    assert task.capability_requirements == {"machine": 2}
    assert task.objective_weights == {
        "time": 0.7,
        "throughput": 0.1,
        "cost": 0.1,
        "stability": 0.1,
    }


def test_resource_has_report_aligned_capabilities_and_reliability():
    resource = Resource(
        "R1",
        "Machine",
        capacity=4.0,
        capabilities={"machine": 3},
        reliability=0.95,
    )

    assert resource.capabilities == {"machine": 3}
    assert resource.reliability == 0.95
