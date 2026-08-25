from src.experiments.alibaba_trace import AlibabaTrace
from src.experiments.run_trace_ppo import run_trace_training
from src.models.resource import Resource
from src.models.task import Task


def test_trace_ppo_runner_records_real_trace_provenance(tmp_path):
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )

    result = run_trace_training(trace, episodes=2, max_steps=3, k_epochs=1, output_dir=tmp_path)

    assert result["data_source"] == "alibaba_cluster_trace_v2018"
    assert len(result["episodes"]) == 2
    assert (tmp_path / "training_log.json").exists()
