import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

from src.models.resource import Resource
from src.models.task import Task


MACHINE_META_COLUMNS = [
    "machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
    "cpu_num", "mem_size", "status",
]
BATCH_TASK_COLUMNS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem",
]


@dataclass
class AlibabaTrace:
    tasks: List[Task]
    resources: List[Resource]
    metadata: Dict[str, str]


def _read_rows(
    path: Union[str, Path],
    columns: List[str],
    delimiter: str = ",",
    limit: int = None,
) -> Iterable[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        yielded = 0
        for values in reader:
            if not values or all(not value.strip() for value in values):
                continue
            if values[0].strip() == columns[0]:
                continue
            if len(values) < len(columns):
                raise ValueError(f"{path} row has {len(values)} columns; expected {len(columns)}")
            yield dict(zip(columns, [value.strip() for value in values]))
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _task_number(task_name: str):
    match = re.match(r"^[A-Za-z]*(\d+)", task_name)
    return int(match.group(1)) if match else None


def _status_reliability(status: str) -> float:
    return 1.0 if status.upper() in {"ONLINE", "RUNNING", "IDLE", "ACTIVE"} else 0.0


def load_v2018_rows(
    machine_meta_path: Union[str, Path],
    batch_task_path: Union[str, Path],
    delimiter: str = ",",
    limit_tasks: int = None,
    limit_resources: int = None,
) -> AlibabaTrace:
    machine_rows = list(_read_rows(machine_meta_path, MACHINE_META_COLUMNS, delimiter, limit_resources))
    task_rows = list(_read_rows(batch_task_path, BATCH_TASK_COLUMNS, delimiter, limit_tasks))

    resources = [
        Resource(
            resource_id=row["machine_id"],
            resource_type="Machine",
            capacity=_float(row["cpu_num"], 1.0),
            capabilities={"cpu": _float(row["cpu_num"]), "memory": _float(row["mem_size"])},
            reliability=_status_reliability(row["status"]),
        )
        for row in machine_rows
    ]

    task_ids = {}
    for row in task_rows:
        number = _task_number(row["task_name"])
        if number is not None:
            task_ids[(row["job_name"], number)] = f"{row['job_name']}:{row['task_name']}"

    tasks = []
    for row in task_rows:
        task_id = f"{row['job_name']}:{row['task_name']}"
        dependencies = []
        suffix_numbers = [int(value) for value in re.findall(r"_(\d+)", row["task_name"])]
        for number in suffix_numbers:
            dependency_id = task_ids.get((row["job_name"], number))
            if dependency_id:
                dependencies.append(dependency_id)
        start = _float(row["start_time"])
        end = _float(row["end_time"], start)
        tasks.append(
            Task(
                task_id=task_id,
                priority=1,
                duration=max(0.001, end - start),
                arrival_time=start,
                dependencies=dependencies,
                capability_requirements={
                    "cpu": _float(row["plan_cpu"]) / 100.0,
                    "memory": _float(row["plan_mem"]),
                },
            )
        )
    return AlibabaTrace(
        tasks=tasks,
        resources=resources,
        metadata={
            "data_source": "alibaba_cluster_trace_v2018",
            "schema": "cluster-trace-v2018/schema.txt",
            "mapping_note": "batch_task and machine_meta mapped to T/R; observed end_time is used only for duration",
        },
    )
