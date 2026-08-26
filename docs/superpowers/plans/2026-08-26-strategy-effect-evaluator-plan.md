# Strategy Effect Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a context-conditioned multi-objective candidate-strategy effect evaluator that generates counterfactual labels, applies feasibility/Pareto/risk selection, validates predictions, and only then supplies demonstrations for a fresh PPO training run.

**Architecture:** A rollout data generator will replay every candidate strategy from the same state and record a fixed-horizon outcome vector plus feasibility and next-state summaries. A PyTorch multi-head predictor will learn those outcomes from `(system, tasks, demands, resources, weights, strategy_id)` and emit mean predictions and uncertainty estimates. A separate decision layer will filter infeasible actions, remove Pareto-dominated actions, enforce risk limits, and use task preferences only as the final tie-breaker.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, Gymnasium, SimPy, pytest, existing scheduling and Alibaba trace adapters.

## Global Constraints

- Keep raw Alibaba trace files outside Git; commit only code, metadata, and small derived summaries.
- Do not reuse old PPO checkpoints after the task-demand observation change; retrain from a new checkpoint.
- Preserve the existing fixed-strategy evaluator and metrics schema for regression compatibility.
- Every new behavior must have a failing test before implementation and pass the full test suite before commit.
- Completion time, failure risk, and recovery time are minimized; throughput is maximized; utilization is treated as a bounded target rather than blindly maximized.

### Task 1: Define the counterfactual outcome schema

**Files:**
- Create: `src/experiments/effect_schema.py`
- Test: `tests/test_effect_schema.py`

**Interfaces:**
- `EFFECT_KEYS = ("completion_time", "throughput", "resource_utilization", "failure_risk", "deadline_risk", "recovery_time")`
- `OutcomeRecord` dataclass with `strategy_id`, `feasible`, `metrics`, `next_state`, and `uncertainty_target` fields.
- `normalize_outcome(metrics, bounds) -> dict[str, float]` with explicit direction handling.

- [ ] Write tests for stable key order, minimization/maximization direction, bounded utilization penalty, and missing optional deadline behavior.
- [ ] Run `D:\Intelligent_Scheduling\.venv\Scripts\python.exe -m pytest tests/test_effect_schema.py -q` and verify failure because the module is absent.
- [ ] Implement the schema and normalization helpers without coupling them to PPO.
- [ ] Run the focused test and then the existing metrics tests.
- [ ] Commit `feat: define strategy effect outcome schema`.

### Task 2: Generate multi-step counterfactual rollout data

**Files:**
- Create: `src/experiments/effect_dataset.py`
- Modify: `src/environment/gym_wrapper.py` only if a state snapshot helper is needed
- Test: `tests/test_effect_dataset.py`

**Interfaces:**
- `generate_counterfactual_records(env_factory, observations, strategy_ids, horizon=3) -> list[OutcomeRecord]`
- `save_effect_dataset(records, output_path) -> Path`
- `load_effect_dataset(path) -> list[OutcomeRecord]`

- [ ] Write a test that replays two candidate actions from the same deterministic state and verifies separate records, identical initial state fingerprints, and bounded horizon.
- [ ] Run the focused test and verify failure before implementation.
- [ ] Implement replay from a copied environment/history, collect per-candidate metrics, capture feasibility, and record the next-state summary.
- [ ] Add JSON serialization with data source, seed, window, candidate set, horizon, and schema version metadata.
- [ ] Run focused tests and a small synthetic dataset generation command.
- [ ] Commit `feat: generate counterfactual strategy effect data`.

### Task 3: Add feasibility, Pareto, and risk-constrained selection

**Files:**
- Create: `src/experiments/effect_selection.py`
- Test: `tests/test_effect_selection.py`

**Interfaces:**
- `filter_feasible(records) -> list[OutcomeRecord]`
- `pareto_front(records, objective_directions) -> list[OutcomeRecord]`
- `select_risk_constrained(records, demand, risk_limits) -> OutcomeRecord`

- [ ] Write tests for infeasible-resource removal, Pareto dominance, and risk-limit rejection even when the rejected strategy has lower completion time.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement strict feasibility filtering, Pareto dominance across completion/throughput/failure/recovery, and final preference tie-breaking from the task-demand vector.
- [ ] Use explicit risk limits such as `failure_risk <= 0.05` and `deadline_risk <= 0.10`; if no record satisfies limits, select the lowest normalized risk violation before preference scoring.
- [ ] Run focused tests and add a synthetic example to the experiment docs.
- [ ] Commit `feat: add pareto and risk constrained strategy selection`.

### Task 4: Train the context-conditioned effect predictor

**Files:**
- Create: `src/models/effect_predictor.py`
- Create: `src/experiments/train_effect_predictor.py`
- Test: `tests/test_effect_predictor.py`

**Interfaces:**
- `EffectPredictor(state_dim, strategy_count, output_dim=6)`
- `predict_distribution(batch) -> (mean, log_variance)`
- `train_effect_predictor(dataset_path, output_dir, epochs, seed) -> dict`
- `evaluate_effect_predictor(model_path, dataset_path) -> dict`

- [ ] Write tests for output shape, finite uncertainty, deterministic inference under a fixed seed, and save/load parity.
- [ ] Run the focused tests and verify failure before implementation.
- [ ] Implement a shared state encoder for system/task-demand/resource/weight features, a strategy embedding, and two output heads for outcome mean and log variance.
- [ ] Train with Gaussian negative log-likelihood for continuous outcomes and report MAE/RMSE, strategy-ranking accuracy, and top-1 selection agreement against rollout labels.
- [ ] Run a small synthetic training smoke test and reject models with non-finite loss or worse-than-baseline ranking accuracy.
- [ ] Commit `feat: train context conditioned effect predictor`.

### Task 5: Produce evaluator-driven demonstrations and validate on held-out scenarios

**Files:**
- Create: `src/experiments/effect_demonstrations.py`
- Modify: `src/experiments/run_trace_ppo.py` only after predictor validation
- Test: `tests/test_effect_demonstrations.py`
- Modify: `docs/experiments/README.md`

**Interfaces:**
- `generate_evaluator_demonstrations(dataset, predictor, selector) -> list[tuple[observation, action, metadata]]`
- Metadata must include `label_type="effect_model"`, candidate predictions, selected strategy, Pareto set, and risk limits.

- [ ] Write tests that generated labels come from the selector and preserve predictor metadata.
- [ ] Run focused tests and verify failure.
- [ ] Implement evaluator-driven labels only after feasibility, Pareto, and risk checks.
- [ ] Split synthetic train/validation/test seeds and Alibaba windows; do not train on evaluation windows.
- [ ] Record model error, top-1 agreement, Pareto recall, and risk violation rate.
- [ ] Commit `feat: generate evaluator driven demonstrations`.

### Task 6: Retrain PPO and compare against fixed strategies

**Files:**
- Modify: `src/experiments/run_trace_ppo.py`
- Modify: `src/experiments/run_ppo.py`
- Create: `src/experiments/evaluate_multimetric.py`
- Test: `tests/test_multimetric_evaluation.py`
- Modify: `docs/midterm_defense.md`

**Interfaces:**
- `evaluate_methods(results, baseline_results, metric_directions) -> dict`
- Report both per-metric deltas and composite constrained decisions; never replace the four primary metrics with one score.

- [ ] Write tests for direction-aware metric deltas and rejection when failure/deadline constraints regress.
- [ ] Run focused tests and verify failure.
- [ ] Retrain PPO from a fresh checkpoint using evaluator-driven demonstrations and the new `demands` observation.
- [ ] Evaluate fixed C01/C03/C04/C05/C09, old PPO baseline if available, and new PPO on identical seeds/windows.
- [ ] Run full tests, compileall, and held-out evaluation.
- [ ] Update the report with the evaluator model definition, validation metrics, and honest limitations.
- [ ] Commit `feat: retrain ppo with validated effect evaluator`.

## Execution Order

Tasks 1-3 establish and test the ground-truth decision layer. Task 4 is blocked until the rollout schema and selector are stable. Task 5 is blocked until predictor ranking and risk validation pass. Task 6 is blocked until held-out predictor validation passes; no new PPO performance claim is made before that gate.
