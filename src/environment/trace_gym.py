import copy

from src.environment.gym_wrapper import SchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace


class TraceSchedulingEnv(SchedulingEnv):
    """Gym environment that replays a selected Alibaba trace workload."""

    def __init__(self, trace: AlibabaTrace, max_queue_size: int = 10, max_resource_size: int = 4):
        self.trace = trace
        super().__init__(
            max_queue_size=max_queue_size,
            max_resource_size=max_resource_size,
            num_strategies=12,
        )

    def _seed_tasks(self):
        for resource in copy.deepcopy(self.trace.resources):
            self.sim.add_resource(resource)
        tasks = copy.deepcopy(self.trace.tasks)
        origin = min((task.arrival_time for task in tasks), default=0.0)
        for task in tasks:
            task.arrival_time -= origin
            self.sim.add_task(task)
        if self.sim.event_queue:
            self.sim.step()
