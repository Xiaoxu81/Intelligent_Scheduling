from typing import Dict

from src.models.task import TaskStatus


CORE_METRICS = (
    "average_completion_time",
    "deadline_satisfaction_rate",
    "throughput",
    "resource_utilization",
    "failure_rate",
    "starvation_risk",
    "recovery_time",
    "decision_time",
)


def empty_metrics() -> Dict[str, float]:
    return {key: 0.0 for key in CORE_METRICS}


def collect_metrics(simulation) -> Dict[str, float]:
    metrics = empty_metrics()
    tasks = list(simulation.tasks.values())
    completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
    failed = [task for task in tasks if task.status == TaskStatus.FAILED]
    if completed:
        completion_times = [task.end_time - task.arrival_time for task in completed]
        metrics["average_completion_time"] = sum(completion_times) / len(completion_times)
        metrics["throughput"] = len(completed) / max(float(simulation.current_time), 1.0)
        metrics["deadline_satisfaction_rate"] = sum(
            task.deadline is None or task.end_time <= task.deadline for task in completed
        ) / len(completed)
    metrics["failure_rate"] = len(failed) / len(tasks) if tasks else 0.0
    fault_times = {}
    recovery_durations = []
    for event in getattr(simulation, "event_history", []):
        if event.event_type == "RESOURCE_FAULT":
            fault_times[event.data] = event.time
        elif event.event_type == "RESOURCE_RECOVERY" and event.data in fault_times:
            recovery_durations.append(event.time - fault_times.pop(event.data))
    metrics["recovery_time"] = sum(recovery_durations) / len(recovery_durations) if recovery_durations else 0.0
    metrics["decision_time"] = (
        simulation.decision_time_total / simulation.decision_count
        if getattr(simulation, "decision_count", 0) else 0.0
    )
    if tasks:
        waiting = [
            max(0.0, simulation.current_time - task.arrival_time)
            for task in tasks
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        metrics["starvation_risk"] = max(waiting, default=0.0)
    elapsed = max(float(simulation.current_time), 1.0)
    capacity = max(len(simulation.resources), 1)
    busy_time = sum(task.cpu_time_used for task in tasks)
    metrics["resource_utilization"] = min(1.0, busy_time / (elapsed * capacity))
    return metrics
