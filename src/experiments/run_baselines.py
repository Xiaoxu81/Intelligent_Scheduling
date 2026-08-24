import argparse
import json
from pathlib import Path

from src.experiments.evaluator import evaluate_candidates
from src.experiments.io import write_experiment_result
from src.experiments.scenarios import ScenarioConfig


DEFAULT_STRATEGIES = ["C01", "C03", "C04", "C05", "C09"]


def run(args=None):
    parser = argparse.ArgumentParser(description="Run report-aligned candidate strategy baselines.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--resources", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results/baselines")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parsed = parser.parse_args(args)

    config = ScenarioConfig(seed=parsed.seed, num_tasks=parsed.tasks, num_resources=parsed.resources)
    results = evaluate_candidates(config, parsed.strategies, repeats=parsed.repeats)
    output_dir = Path(parsed.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {result.strategy_id: result.metrics for result in results}
    paths = write_experiment_result(
        output_dir,
        metadata={"seed": parsed.seed, "tasks": parsed.tasks, "resources": parsed.resources, "repeats": parsed.repeats},
        task_rows=[],
        summary=summary,
    )
    (output_dir / "strategies.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return results


if __name__ == "__main__":
    run()
