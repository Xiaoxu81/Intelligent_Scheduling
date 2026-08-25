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

The plotting helpers create comparison and PPO reward PNGs from the saved JSON files.

The research design and implementation plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
