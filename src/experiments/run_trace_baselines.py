import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, Optional, Union

from src.environment.simulation import SimulationEnv
from src.experiments.alibaba_trace import AlibabaTrace, load_v2018_rows
from src.experiments.io import write_experiment_result
from src.experiments.metrics import collect_metrics
from src.strategies.heuristics import get_scheduler_by_id


DEFAULT_STRATEGIES = ["C01", "C03", "C04", "C05", "C09"]


def _bounded_trace(trace: AlibabaTrace, limit_tasks: Optional[int], limit_resources: Optional[int]) -> AlibabaTrace:
    tasks = trace.tasks[:limit_tasks] if limit_tasks else trace.tasks[:]
    task_ids = {task.task_id for task in tasks}
    for task in tasks:
        task.dependencies = [dependency for dependency in task.dependencies if dependency in task_ids]
    resources = trace.resources[:limit_resources] if limit_resources else trace.resources[:]
    metadata = dict(trace.metadata)
    metadata.update({"selected_tasks": len(tasks), "selected_resources": len(resources)})
    return AlibabaTrace(tasks=tasks, resources=resources, metadata=metadata)


def _run_one(
    trace: AlibabaTrace,
    strategy_id: str,
    fault_resource: Optional[str] = None,
    fault_at: Optional[float] = None,
    fault_duration: float = 0.0,
):
    env = SimulationEnv(scheduler=get_scheduler_by_id(strategy_id))
    for resource in copy.deepcopy(trace.resources):
        env.add_resource(resource)
    tasks = copy.deepcopy(trace.tasks)
    time_origin = min((task.arrival_time for task in tasks), default=0.0)
    for task in tasks:
        task.arrival_time -= time_origin
        task.wait_start_time = task.arrival_time
        if task.deadline is not None:
            task.deadline -= time_origin
        env.add_task(task)
    if fault_resource and fault_at is not None:
        if fault_resource not in env.resources:
            raise ValueError(f"fault_resource {fault_resource!r} is not in selected resources")
        env.inject_resource_fault(
            fault_resource,
            delay=max(0.0, fault_at - time_origin),
            duration=max(0.0, fault_duration),
        )
    while env.event_queue:
        env.step()
    metrics = collect_metrics(env)
    task_rows = [
        {
            "strategy_id": strategy_id,
            "task_id": task.task_id,
            "status": task.status.value,
            "arrival_time": task.arrival_time,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "assigned_resource_id": task.assigned_resource_id,
            "duration": task.duration,
        }
        for task in env.tasks.values()
    ]
    return metrics, task_rows


def run_trace_baselines(
    machine_meta: Union[str, Path],
    batch_task: Union[str, Path],
    output: Union[str, Path],
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    limit_tasks: Optional[int] = None,
    limit_resources: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit_jobs: Optional[int] = None,
    complete_jobs: bool = True,
    fault_resource: Optional[str] = None,
    fault_at: Optional[float] = None,
    fault_duration: float = 0.0,
):
    strategies = list(strategies)
    trace = _bounded_trace(
        load_v2018_rows(
            machine_meta,
            batch_task,
            limit_tasks=limit_tasks,
            limit_resources=limit_resources,
            start_time=start_time,
            end_time=end_time,
            limit_jobs=limit_jobs,
            complete_jobs=complete_jobs,
        ),
        limit_tasks=None,
        limit_resources=None,
    )
    trace.metadata["time_origin"] = min((task.arrival_time for task in trace.tasks), default=0.0)
    summary = {}
    task_rows = []
    for strategy_id in strategies:
        metrics, rows = _run_one(
            trace,
            strategy_id,
            fault_resource=fault_resource,
            fault_at=fault_at,
            fault_duration=fault_duration,
        )
        summary[strategy_id] = metrics
        task_rows.extend(rows)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **trace.metadata,
        "machine_meta_path": str(machine_meta),
        "batch_task_path": str(batch_task),
        "strategies": list(strategies),
        "experiment": "trace_baseline",
        "simulation_engine": "src.environment.simulation.SimulationEnv",
        "raw_data_committed": False,
        "fault_profile": (
            "trace_workload_controlled_fault" if fault_resource and fault_at is not None else "none"
        ),
        "fault_resource": fault_resource,
        "fault_at": fault_at,
        "fault_duration": fault_duration if fault_resource and fault_at is not None else None,
    }
    paths = write_experiment_result(output_dir, metadata, task_rows, summary)
    (output_dir / "strategies.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def run(args=None):
    parser = argparse.ArgumentParser(description="Run strategies on an Alibaba Cluster Trace v2018 subset.")
    parser.add_argument("--machine-meta", required=True)
    parser.add_argument("--batch-task", required=True)
    parser.add_argument("--output", default="results/trace-baselines")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--limit-resources", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--limit-jobs", type=int)
    parser.add_argument("--partial-jobs", action="store_true")
    parser.add_argument("--fault-resource")
    parser.add_argument("--fault-at", type=float)
    parser.add_argument("--fault-duration", type=float, default=0.0)
    parsed = parser.parse_args(args)
    paths = run_trace_baselines(
        machine_meta=parsed.machine_meta,
        batch_task=parsed.batch_task,
        output=parsed.output,
        strategies=parsed.strategies,
        limit_tasks=parsed.limit_tasks,
        limit_resources=parsed.limit_resources,
        start_time=parsed.start_time,
        end_time=parsed.end_time,
        limit_jobs=parsed.limit_jobs,
        complete_jobs=not parsed.partial_jobs,
        fault_resource=parsed.fault_resource,
        fault_at=parsed.fault_at,
        fault_duration=parsed.fault_duration,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


if __name__ == "__main__":
    run()
