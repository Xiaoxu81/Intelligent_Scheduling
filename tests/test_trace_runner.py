from src.experiments.run_trace_baselines import run_trace_baselines


def test_trace_runner_writes_provenance_and_strategy_outputs(tmp_path):
    machine_path = tmp_path / "machine_meta.csv"
    task_path = tmp_path / "batch_task.csv"
    machine_path.write_text(
        "machine_id,time_stamp,failure_domain_1,failure_domain_2,cpu_num,mem_size,status\n"
        "M1,0,fd-a,fd-b,8,64,ONLINE\n",
        encoding="utf-8",
    )
    task_path.write_text(
        "task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem\n"
        "task1,1,jobA,M,Terminated,0,2,100,10\n"
        "task2_1,1,jobA,M,Terminated,2,5,100,10\n",
        encoding="utf-8",
    )

    output = tmp_path / "results"
    run_trace_baselines(
        machine_meta=machine_path,
        batch_task=task_path,
        output=output,
        strategies=["C01"],
        limit_tasks=2,
        limit_resources=1,
    )

    metadata = (output / "result.json").read_text(encoding="utf-8")
    assert "alibaba_cluster_trace_v2018" in metadata
    assert (output / "strategies.json").exists()
    assert (output / "tasks.csv").exists()


def test_trace_runner_records_controlled_fault_metrics(tmp_path):
    machine_path = tmp_path / "machine_meta.csv"
    task_path = tmp_path / "batch_task.csv"
    machine_path.write_text(
        "M1,0,fd-a,fd-b,8,64,ONLINE\n",
        encoding="utf-8",
    )
    task_path.write_text(
        "task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem\n"
        "task1,1,jobA,M,Terminated,0,5,100,10\n",
        encoding="utf-8",
    )

    output = tmp_path / "fault-results"
    run_trace_baselines(
        machine_meta=machine_path,
        batch_task=task_path,
        output=output,
        strategies=["C01"],
        fault_resource="M1",
        fault_at=1.0,
        fault_duration=2.0,
    )

    payload = __import__("json").loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["fault_profile"] == "trace_workload_controlled_fault"
    assert payload["summary"]["C01"]["failure_rate"] > 0
