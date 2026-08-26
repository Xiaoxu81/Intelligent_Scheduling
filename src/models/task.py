from enum import Enum
from typing import Dict, List, Optional

from src.models.task_demand import derive_task_demand

class TaskStatus(Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Task:
    def __init__(
        self,
        task_id: str,
        priority: int,
        duration: float,
        arrival_time: float = 0.0,
        deadline: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        capability_requirements: Optional[Dict[str, float]] = None,
        objective_weights: Optional[Dict[str, float]] = None,
        failure_penalty: float = 0.0,
        cost_sensitivity: float = 0.5,
        stability_requirement: float = 0.5,
    ):
        self.task_id = task_id
        self.priority = priority
        self.duration = duration
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.capability_requirements = dict(capability_requirements or {})
        self.failure_penalty = float(failure_penalty)
        self.cost_sensitivity = float(cost_sensitivity)
        self.stability_requirement = float(stability_requirement)
        self.objective_weights = {
            "time": 0.25,
            "throughput": 0.25,
            "cost": 0.25,
            "stability": 0.25,
        }
        if objective_weights:
            self.objective_weights.update(objective_weights)
        
        self.is_realtime = False  # 默认为非实时任务
        self.cpu_time_used = 0.0  # 已使用的 CPU 时间
        self.wait_start_time: Optional[float] = arrival_time # 开始等待的时间
        
        self.status = TaskStatus.PENDING
        self.remaining_time = duration
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.assigned_resource_id: Optional[str] = None

    def demand_features(
        self,
        current_time: float = 0.0,
        downstream_count: int = 0,
        feasible_resource_count: int = 0,
        total_resource_count: int = 0,
    ) -> Dict[str, object]:
        """Return state-aware demand scores and derived objective weights."""
        return derive_task_demand(
            priority=self.priority,
            duration=self.remaining_time,
            arrival_time=self.arrival_time,
            current_time=current_time,
            deadline=self.deadline,
            dependency_count=len(self.dependencies),
            downstream_count=downstream_count,
            failure_penalty=self.failure_penalty,
            feasible_resource_count=feasible_resource_count,
            total_resource_count=total_resource_count,
            cost_sensitivity=self.cost_sensitivity,
            stability_requirement=self.stability_requirement,
        )

    def __repr__(self):
        return f"Task(id={self.task_id}, priority={self.priority}, status={self.status.value})"
