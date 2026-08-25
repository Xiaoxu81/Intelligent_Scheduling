import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, Union


def aggregate_strategy_files(paths: Iterable[Union[str, Path]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(payload)
    methods = sorted({method for row in rows for method in row})
    metrics = sorted({metric for row in rows for method in row for metric in row[method]})
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        result[method] = {}
        for metric in metrics:
            values = [row[method][metric] for row in rows if method in row and metric in row[method]]
            if not values:
                continue
            result[method][metric] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return result


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate strategy result JSON files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/aggregate.json"))
    parsed = parser.parse_args(args)
    summary = aggregate_strategy_files(parsed.files)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(parsed.output)


if __name__ == "__main__":
    main()
