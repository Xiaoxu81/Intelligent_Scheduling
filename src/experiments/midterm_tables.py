"""Build reproducible midterm-defense tables from saved experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRICS = ("average_completion_time", "throughput", "resource_utilization", "failure_rate")


def _read(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _window_key(window: list[int] | tuple[int, int]) -> str:
    return f"{window[0]}-{window[1]}"


def _metrics(record: dict[str, Any]) -> dict[str, float]:
    source = record.get("metrics")
    if source is None:
        source = record.get("evaluation", {}).get("metrics")
    if source is None:
        source = record
    return {name: float(source.get(name, 0.0)) for name in METRICS}


def build_midterm_summary(
    fixed_path: str | Path | list[str | Path],
    ppo_path: str | Path,
    fault_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Return the four tables required by the midterm report.

    Completion time is minimized. Improvement is therefore computed as
    ``(best_fixed - ppo) / best_fixed * 100``.
    """
    fixed = {}
    fixed_paths = fixed_path if isinstance(fixed_path, list) else [fixed_path]
    for path in fixed_paths:
        loaded = _read(path)
        for key, value in loaded.items():
            if isinstance(value, dict) and key in fixed and isinstance(fixed[key], dict):
                fixed[key].update(value)
            else:
                fixed[key] = value
    ppo = _read(ppo_path)
    # Older fine-grained baseline artifacts use semantic aliases instead of
    # trace-window keys. Keep the mapping explicit so the report cannot
    # accidentally compare a PPO window with another window's baseline.
    if "formal" in fixed:
        fixed.setdefault("250000-260000", fixed["formal"])
    if "independent" in fixed:
        fixed.setdefault("260000-270000", fixed["independent"])
    faults = _read(fault_path)
    fixed_vs_ppo: list[dict[str, Any]] = []
    metrics_table: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for item in ppo:
        window = item.get("window")
        key = _window_key(window) if window else item.get("window_key", "unknown")
        ppo_metrics = _metrics(item)
        fixed_for_window = fixed.get(key, fixed.get("formal", {}))
        if not fixed_for_window:
            continue
        ordered_strategies = sorted(fixed_for_window, key=lambda name: fixed_for_window[name]["average_completion_time"])
        best_strategy = ordered_strategies[0]
        best = _metrics(fixed_for_window[best_strategy])
        for strategy in ordered_strategies:
            fixed_vs_ppo.append({"window": key, "strategy": strategy, "metrics": _metrics(fixed_for_window[strategy]), "ppo": ppo_metrics, "is_best": strategy == best_strategy})
        metrics_table.append({"window": key, "method": "PPO", **ppo_metrics})
        for strategy in ordered_strategies:
            metrics_table.append({"window": key, "method": f"fixed:{strategy}", **_metrics(fixed_for_window[strategy])})
        improvement = (best["average_completion_time"] - ppo_metrics["average_completion_time"]) / best["average_completion_time"] * 100
        improvements.append({"window": key, "best_fixed_strategy": best_strategy, "ppo_average_completion_time": ppo_metrics["average_completion_time"], "best_fixed_average_completion_time": best["average_completion_time"], "improvement_percent": improvement})

    normal_by_window = {item.get("window_key", _window_key(item["window"])): item for item in ppo if item.get("scenario", "normal") == "normal"}
    normal_vs_fault = []
    for item in faults:
        if item.get("scenario", "normal") == "normal":
            continue
        key = item.get("window_key", _window_key(item["window"]))
        normal = normal_by_window.get(key)
        normal_vs_fault.append({"window": key, "scenario": item.get("scenario", "fault"), "normal": _metrics(normal) if normal else None, "fault": _metrics(item)})

    return {"fixed_vs_ppo": fixed_vs_ppo, "metrics": metrics_table, "normal_vs_fault": normal_vs_fault, "improvement": improvements}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(summary: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# 中期答辩实验结果汇总", "", "说明：平均完成时间越低越好；吞吐量、资源利用率越高越好；失败率越低越好。Alibaba trace 的 deadline 指标不作为主要结论。", ""]
    lines += ["## 1. 固定策略与 PPO 对比", "", "| 窗口 | C01 | C03 | C04 | C05 | C09 | PPO |", "|---|---:|---:|---:|---:|---:|---:|"]
    grouped: dict[str, dict[str, Any]] = {}
    for row in summary["fixed_vs_ppo"]:
        grouped.setdefault(row["window"], {})[row["strategy"]] = row["metrics"]["average_completion_time"]
        grouped[row["window"]]["PPO"] = row["ppo"]["average_completion_time"]
    for window, values in grouped.items():
        lines.append("| " + window + " | " + " | ".join(_fmt(values.get(name)) for name in ("C01", "C03", "C04", "C05", "C09", "PPO")) + " |")
    lines += ["", "## 2. 平均完成时间、吞吐量、资源利用率、失败率", "", "| 窗口 | 方法 | 平均完成时间 | 吞吐量 | 资源利用率 | 失败率 |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        lines.append(f"| {row['window']} | {row['method']} | {_fmt(row['average_completion_time'])} | {_fmt(row['throughput'])} | {_fmt(row['resource_utilization'])} | {_fmt(row['failure_rate'])} |")
    lines += ["", "## 3. 正常场景与故障场景", "", "| 窗口 | 场景 | 正常平均完成时间 | 故障平均完成时间 | 故障失败率 |", "|---|---|---:|---:|---:|"]
    for row in summary["normal_vs_fault"]:
        normal = row["normal"] or {}
        fault = row["fault"]
        lines.append(f"| {row['window']} | {row['scenario']} | {_fmt(normal.get('average_completion_time'))} | {_fmt(fault['average_completion_time'])} | {_fmt(fault['failure_rate'])} |")
    lines += ["", "## 4. PPO 相比最优固定策略的提升百分比", "", "| 窗口 | 最优固定策略 | PPO 平均完成时间 | 最优固定平均完成时间 | 提升百分比 |", "|---|---|---:|---:|---:|"]
    for row in summary["improvement"]:
        lines.append(f"| {row['window']} | {row['best_fixed_strategy']} | {_fmt(row['ppo_average_completion_time'])} | {_fmt(row['best_fixed_average_completion_time'])} | {_fmt(row['improvement_percent'])}% |")
    return "\n".join(lines) + "\n"


def build_and_write(output_dir: str | Path, fixed_path: str | Path, ppo_path: str | Path, fault_path: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_midterm_summary(fixed_path, ppo_path, fault_path)
    (output / "midterm_tables.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "midterm_tables.md").write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    build_and_write("results/midterm", ["results/trace-four-windows/baselines-new.json", "results/trace-mixed/fine-grained-baselines.json"], "results/trace-four-windows/ppo-regularized/evaluation.json", "results/trace-mixed-fault/evaluation.json")
