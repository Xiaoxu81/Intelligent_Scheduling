import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union


def write_experiment_result(
    directory: Union[str, Path],
    metadata: Mapping[str, Any],
    task_rows: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Path]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = [dict(row) for row in task_rows]
    payload = {
        "metadata": dict(metadata),
        "summary": dict(summary),
        "tasks": task_rows,
    }
    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "tasks.csv"
    fieldnames = sorted({key for row in task_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(task_rows)
    return {"json": json_path, "tasks_csv": csv_path}
