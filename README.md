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

The research design and implementation plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
