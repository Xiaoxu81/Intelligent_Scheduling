import copy
from typing import Iterable, Optional

from src.environment.gym_wrapper import SchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace


class TraceSchedulingEnv(SchedulingEnv):
    """Gym environment that replays a selected Alibaba trace workload."""

    def __init__(
        self,
        trace: AlibabaTrace,
        max_queue_size: int = 10,
        max_resource_size: int = 4,
        strategy_ids: Optional[Iterable[str]] = None,
        max_assignments_per_step: Optional[int] = None,
        fault_resource: Optional[str] = None,
        fault_at: Optional[float] = None,
        fault_duration: float = 0.0,
    ):
        self.trace = trace
        if (fault_resource is None) != (fault_at is None):
            raise ValueError("fault_resource and fault_at must be provided together")
        if fault_at is not None and (fault_at < 0 or fault_duration <= 0):
            raise ValueError("fault_at must be non-negative and fault_duration must be positive")
        self.fault_resource = fault_resource
        self.fault_at = fault_at
        self.fault_duration = fault_duration
        requested_strategy_ids = list(strategy_ids or [f"C{i:02d}" for i in range(1, 13)])
        if not requested_strategy_ids or any(
            strategy_id not in {f"C{i:02d}" for i in range(1, 13)} for strategy_id in requested_strategy_ids
        ):
            raise ValueError("strategy_ids must contain valid C01-C12 identifiers")
        super().__init__(
            max_queue_size=max_queue_size,
            max_resource_size=max_resource_size,
            num_strategies=len(requested_strategy_ids),
            max_assignments_per_step=max_assignments_per_step,
        )
        self.strategy_ids = requested_strategy_ids

    def _seed_tasks(self):
        for resource in copy.deepcopy(self.trace.resources):
            self.sim.add_resource(resource)
        tasks = copy.deepcopy(self.trace.tasks)
        origin = min((task.arrival_time for task in tasks), default=0.0)
        for task in tasks:
            task.arrival_time -= origin
            self.sim.add_task(task)
        if self.fault_resource is not None:
            if self.fault_resource not in self.sim.resources:
                raise ValueError(f"fault_resource {self.fault_resource!r} is not in trace resources")
            self.sim.inject_resource_fault(self.fault_resource, self.fault_at, self.fault_duration)
        if self.sim.event_queue:
            self.sim.step()
