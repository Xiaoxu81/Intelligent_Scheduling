# Scheduling Experiment Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the existing scheduling prototype with the midterm-report methodology and produce reproducible baseline, evaluation, and PPO-ready experiment data.

**Architecture:** Preserve the SimPy event engine and existing strategy interfaces. Add structured task/resource requirements, deterministic scenario configuration, a metrics collector, a same-scenario candidate-strategy evaluator, and serializable experiment outputs. Extend the Gym observation only after the baseline evaluator is validated.

**Tech Stack:** Python 3, SimPy, Gymnasium, NumPy, PyTorch, JSON/CSV, unittest-compatible tests.

## Global Constraints

- Use the midterm-report variables `S`, `T`, `R`, and `wT` as the research-facing state definition.
- Keep the existing 12 candidate strategies interpretable and consistently identified.
- Compare strategies on identical scenario definitions and random seeds.
- Do not fabricate experiment results; every reported value must come from saved raw output.
- Preserve existing event behavior unless a failing test demonstrates a required correction.

---

### Task 1: Stabilize and test the existing simulation baseline

**Files:**
- Read: `src/environment/simulation.py`
- Read: `src/environment/gym_wrapper.py`
- Read: `src/strategies/*.py`
- Test: `tests/test_env.py`, `tests/test_gym_env.py`, `tests/test_strategies.py`

**Interfaces:**
- Produces a verified baseline runner contract: a scenario can be reset, executed, and summarized without relying on console output.

- [ ] Run the existing test files directly with the project Python runtime and record failures.
- [ ] Add a minimal regression test for the current end-to-end environment lifecycle.
- [ ] Fix only issues required for deterministic execution and testability.
- [ ] Re-run the baseline tests and confirm the failure cause for any remaining dependency issue.

### Task 2: Extend task and resource domain data for `T`, `R`, and `wT`

**Files:**
- Modify: `src/models/task.py`
- Modify: `src/models/resource.py`
- Test: `tests/test_domain_requirements.py`

**Interfaces:**
- `Task` exposes `capability_requirements` and `objective_weights` with normalized defaults for time, throughput, cost, and stability.
- `Resource` exposes `capabilities`, `capacity`, and `reliability` while retaining existing status and assignment behavior.

- [ ] Write failing tests for default values, explicit values, and serialization-ready access.
- [ ] Run the tests and verify they fail because the fields are absent.
- [ ] Implement the smallest backward-compatible model extension.
- [ ] Run the tests and existing simulation tests.

### Task 3: Add deterministic scenarios and unified metrics

**Files:**
- Create: `src/experiments/scenarios.py`
- Create: `src/experiments/metrics.py`
- Create: `src/experiments/io.py`
- Test: `tests/test_experiment_data.py`

**Interfaces:**
- `ScenarioConfig(seed, num_tasks, num_resources, arrival_profile, fault_profile)` describes a reproducible scenario.
- `build_scenario(config) -> Scenario` returns task/resource/event definitions without running the experiment.
- `collect_metrics(simulation) -> dict[str, float]` returns completion time, deadline satisfaction, throughput, utilization, failure, starvation, recovery time, and decision time fields.
- `write_experiment_result(path, metadata, task_rows, summary)` writes JSON and CSV-compatible records without losing metadata.

- [ ] Write failing tests for same-seed scenario equality and required metric keys.
- [ ] Run the tests and verify they fail for missing experiment modules.
- [ ] Implement deterministic scenario generation using local random generators rather than global mutable state.
- [ ] Implement metric collection from task/resource state and explicit event records.
- [ ] Implement result serialization with seed, method, configuration, raw task rows, and summary.
- [ ] Run focused and existing tests.

### Task 4: Implement same-scenario candidate strategy evaluation

**Files:**
- Create: `src/experiments/evaluator.py`
- Modify: `src/strategies/composite.py` if stable strategy names are missing
- Test: `tests/test_strategy_evaluator.py`

**Interfaces:**
- `evaluate_strategy(config, strategy_id, repeats=1) -> StrategyResult` runs one candidate strategy on reproducible copies of the same scenario.
- `evaluate_candidates(config, strategy_ids, repeats) -> list[StrategyResult]` returns raw metrics for all candidates.
- `select_demonstration_label(results, objective_weights) -> LabelResult` returns the feasible candidate with minimum normalized weighted loss and its component scores.

- [ ] Write failing tests for identical scenario inputs, candidate coverage, infeasible handling, and weighted label selection.
- [ ] Run the tests and verify they fail before implementation.
- [ ] Implement strategy cloning and evaluation without changing the shared original scenario.
- [ ] Implement normalization with recorded min/max or fixed metric direction metadata.
- [ ] Implement weighted loss and deterministic tie-breaking by stable strategy ID.
- [ ] Run focused tests and a small manual baseline experiment.

### Task 5: Generate baseline experiment artifacts

**Files:**
- Create: `src/experiments/run_baselines.py`
- Create: `src/experiments/aggregate.py`
- Test: `tests/test_baseline_outputs.py`

**Interfaces:**
- CLI accepts seed, task/resource counts, repeats, output directory, and strategy IDs.
- Runner writes per-run raw results and aggregate mean/std summaries suitable for report tables.

- [ ] Write failing tests for required output files and aggregate statistics.
- [ ] Run the tests and verify they fail for the missing runner.
- [ ] Implement baseline execution for FCFS, EDF, fixed priority, random, and current PPO-compatible strategy selection where available.
- [ ] Implement aggregation by method and scenario with sample count, mean, and standard deviation.
- [ ] Run a small multi-seed baseline and inspect the generated JSON/CSV.

### Task 6: Align the PPO observation and prepare demonstration training

**Files:**
- Modify: `src/environment/gym_wrapper.py`
- Modify: `src/models/drl_agent.py`
- Create: `src/experiments/demonstrations.py`
- Test: `tests/test_report_state.py`, `tests/test_demonstrations.py`

**Interfaces:**
- Observation contains separately inspectable system, task, resource, and objective-weight components corresponding to `S`, `T`, `R`, and `wT`.
- Demonstration generator writes state vectors, candidate label, weighted scores, and scenario metadata.

- [ ] Write failing tests for observation component shapes and demonstration schema.
- [ ] Run tests and verify they fail for missing report-aligned fields.
- [ ] Extend observation encoding with explicit masks and backward-compatible bounds.
- [ ] Add demonstration generation from evaluator labels.
- [ ] Update PPO input dimensions only after observation tests pass.
- [ ] Run existing Gym tests and a short training smoke test.

### Task 7: Add report-ready plots and experiment documentation

**Files:**
- Create: `src/experiments/plots.py`
- Create: `docs/experiments/README.md`
- Test: `tests/test_experiment_plots.py`

- [ ] Write failing tests for plot input validation and output naming.
- [ ] Implement plots for method comparison, training reward, and ablation summaries from saved data only.
- [ ] Document commands, configuration schema, output layout, and how to trace every reported number to raw data.
- [ ] Run the full available test suite and one end-to-end baseline command.

