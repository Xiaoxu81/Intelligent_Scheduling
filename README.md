# Intelligent_Scheduling
Intelligent_Scheduling
# Intelligent Scheduling

This repository contains the scheduling simulation prototype and the report-aligned experiment pipeline.

## Baseline experiments

Using the project virtual environment, run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.experiments.run_baselines --seed 11 --tasks 6 --resources 2 --repeats 2 --output results/baselines
```

The runner evaluates reproducible candidate strategies and writes JSON/CSV artifacts under `results/`.

To run a short PPO smoke experiment and save the training curve plus model checkpoint:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.experiments.run_ppo --seed 7 --episodes 5 --max-steps 30 --k-epochs 1 --output results/ppo
```

To aggregate several baseline result files:

```powershell
python -m src.experiments.aggregate results/baselines-multiseed/seed1/strategies.json results/baselines-multiseed/seed2/strategies.json --output results/baselines-multiseed/aggregate.json
```

To run the same strategies on a local Alibaba Cluster Trace v2018 subset:

```powershell
python -m src.experiments.run_trace_baselines --machine-meta D:\data\machine_meta.csv --batch-task D:\data\batch_task.csv --limit-tasks 500 --limit-resources 50 --output results/trace-baselines/subset1
```

Raw trace files are intentionally kept outside the repository. See `docs/experiments/README.md` for provenance and interpretation notes.

The plotting helpers create comparison and PPO reward PNGs from the saved JSON files.

The research design and implementation plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
