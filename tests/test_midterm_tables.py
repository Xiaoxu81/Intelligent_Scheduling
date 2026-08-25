import json

from src.experiments.midterm_tables import build_midterm_summary


def test_midterm_summary_contains_four_required_tables(tmp_path):
    fixed = {
        "1-2": {
            "C01": {"average_completion_time": 10, "throughput": 0.1, "resource_utilization": 0.5, "failure_rate": 0.0},
            "C03": {"average_completion_time": 12, "throughput": 0.1, "resource_utilization": 0.5, "failure_rate": 0.0},
        }
    }
    ppo = [{"window": [1, 2], "evaluation": {"metrics": {"average_completion_time": 9, "throughput": 0.1, "resource_utilization": 0.5, "failure_rate": 0.0}}}]
    fault = [{"window": [1, 2], "scenario": "unseen-fault", "evaluation": {"metrics": {"average_completion_time": 11, "throughput": 0.09, "resource_utilization": 0.4, "failure_rate": 0.1}}}]
    fixed_path = tmp_path / "fixed.json"
    ppo_path = tmp_path / "ppo.json"
    fault_path = tmp_path / "fault.json"
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    ppo_path.write_text(json.dumps(ppo), encoding="utf-8")
    fault_path.write_text(json.dumps(fault), encoding="utf-8")

    summary = build_midterm_summary(fixed_path, ppo_path, fault_path)

    assert set(summary) == {"fixed_vs_ppo", "metrics", "normal_vs_fault", "improvement"}
    assert summary["fixed_vs_ppo"][0]["strategy"] == "C01"
    assert summary["improvement"][0]["improvement_percent"] == 10.0
    assert summary["normal_vs_fault"][0]["scenario"] == "unseen-fault"
