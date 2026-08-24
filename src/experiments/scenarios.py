from dataclasses import dataclass, field
import random
from typing import Any, Dict, List

from src.models.resource import Resource
from src.models.task import Task


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 0
    num_tasks: int = 10
    num_resources: int = 3
    arrival_profile: str = "uniform"
    fault_profile: str = "none"


@dataclass
class Scenario:
    config: ScenarioConfig
    tasks: List[Task] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__.copy(),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "priority": task.priority,
                    "duration": task.duration,
                    "arrival_time": task.arrival_time,
                    "deadline": task.deadline,
                    "dependencies": list(task.dependencies),
                    "capability_requirements": dict(task.capability_requirements),
                    "objective_weights": dict(task.objective_weights),
                }
                for task in self.tasks
            ],
            "resources": [
                {
                    "resource_id": resource.resource_id,
                    "resource_type": resource.resource_type,
                    "capacity": resource.capacity,
                    "capabilities": dict(resource.capabilities),
                    "reliability": resource.reliability,
                }
                for resource in self.resources
            ],
            "events": list(self.events),
        }


def build_scenario(config: ScenarioConfig) -> Scenario:
    rng = random.Random(config.seed)
    resources = [
        Resource(
            resource_id=f"R{i + 1}",
            resource_type="Machine",
            capacity=float(rng.randint(1, 4)),
            capabilities={"machine": float(rng.randint(1, 3))},
            reliability=round(rng.uniform(0.85, 1.0), 4),
        )
        for i in range(config.num_resources)
    ]

    tasks: List[Task] = []
    for i in range(config.num_tasks):
        arrival_time = float(i if config.arrival_profile == "staggered" else rng.randint(0, max(0, config.num_tasks // 2)))
        duration = float(rng.randint(1, 8))
        priority = rng.randint(1, 5)
        deadline = arrival_time + duration + float(rng.randint(2, 8))
        dependencies = [f"T{i}" ] if i > 0 and rng.random() < 0.25 else []
        weight_values = [rng.random() for _ in range(4)]
        weight_total = sum(weight_values) or 1.0
        weights = dict(zip(("time", "throughput", "cost", "stability"), [round(v / weight_total, 6) for v in weight_values]))
        tasks.append(
            Task(
                task_id=f"T{i + 1}",
                priority=priority,
                duration=duration,
                arrival_time=arrival_time,
                deadline=deadline,
                dependencies=dependencies,
                capability_requirements={"machine": float(rng.randint(1, 3))},
                objective_weights=weights,
            )
        )

    events: List[Dict[str, Any]] = []
    if config.fault_profile != "none" and resources:
        events.append({"type": "resource_fault", "resource_id": resources[0].resource_id, "delay": 5.0, "duration": 2.0})
    return Scenario(config=config, tasks=tasks, resources=resources, events=events)
