from pathlib import Path

from src.experiments.alibaba_trace import load_v2018_rows


def test_alibaba_v2018_rows_map_tasks_resources_and_dag_dependencies(tmp_path):
    machine_path = tmp_path / "machine_meta.csv"
    task_path = tmp_path / "batch_task.csv"
    machine_path.write_text("M1,0,1,fd-a,8,64,ONLINE\n", encoding="utf-8")
    task_path.write_text(
        "task1,1,jobA,M,Terminated,0,5,200,10\n"
        "M2_1,1,jobA,M,Terminated,5,8,100,5\n",
        encoding="utf-8",
    )

    trace = load_v2018_rows(machine_path, task_path)

    assert len(trace.tasks) == 2
    assert trace.tasks[1].dependencies == ["jobA:task1"]
    assert trace.tasks[0].capability_requirements["cpu"] == 2.0
    assert trace.resources[0].capabilities["cpu"] == 8.0
    assert trace.metadata["data_source"] == "alibaba_cluster_trace_v2018"


def test_alibaba_v2018_loader_can_bound_raw_rows(tmp_path):
    machine_path = tmp_path / "machine_meta.csv"
    task_path = tmp_path / "batch_task.csv"
    machine_path.write_text(
        "M1,0,1,fd-a,8,64,ONLINE\nM2,0,1,fd-b,8,64,ONLINE\n",
        encoding="utf-8",
    )
    task_path.write_text(
        "task1,1,jobA,M,Terminated,0,1,100,10\n"
        "task2,1,jobA,M,Terminated,1,2,100,10\n",
        encoding="utf-8",
    )

    trace = load_v2018_rows(machine_path, task_path, limit_tasks=1, limit_resources=1)

    assert len(trace.tasks) == 1
    assert len(trace.resources) == 1


def test_alibaba_v2018_loader_selects_time_window_and_collapses_instances(tmp_path):
    machine_path = tmp_path / "machine_meta.csv"
    task_path = tmp_path / "batch_task.csv"
    machine_path.write_text("M1,0,1,fd-a,8,64,ONLINE\n", encoding="utf-8")
    task_path.write_text(
        "M1,1,j1,1,Terminated,10,12,100,10\n"
        "M1,2,j1,1,Terminated,10,15,100,10\n"
        "M2_1,1,j1,1,Terminated,15,20,100,10\n"
        "M1,1,j2,1,Terminated,100,101,100,10\n",
        encoding="utf-8",
    )

    trace = load_v2018_rows(
        machine_path,
        task_path,
        start_time=10,
        end_time=21,
        complete_jobs=True,
    )

    assert [task.task_id for task in trace.tasks] == ["j1:M1", "j1:M2_1"]
    assert trace.tasks[0].arrival_time == 10
    assert trace.tasks[0].duration == 5
