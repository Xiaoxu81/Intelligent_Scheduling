from enum import Enum
from typing import Dict, List, Optional

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
    ):
        self.task_id = task_id
        self.priority = priority
        self.duration = duration
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.capability_requirements = dict(capability_requirements or {})
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

    def __repr__(self):
        return f"Task(id={self.task_id}, priority={self.priority}, status={self.status.value})"
