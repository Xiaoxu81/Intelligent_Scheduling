import json

from src.experiments.aggregate import aggregate_strategy_files


def test_aggregate_strategy_files_reports_mean_std_and_sample_count(tmp_path):
    first = tmp_path / "seed1.json"
    second = tmp_path / "seed2.json"
    first.write_text(json.dumps({"C01": {"throughput": 1.0, "failure_rate": 0.0}}), encoding="utf-8")
    second.write_text(json.dumps({"C01": {"throughput": 3.0, "failure_rate": 0.2}}), encoding="utf-8")

    summary = aggregate_strategy_files([first, second])

    assert summary["C01"]["throughput"]["n"] == 2
    assert summary["C01"]["throughput"]["mean"] == 2.0
    assert summary["C01"]["failure_rate"]["mean"] == 0.1
