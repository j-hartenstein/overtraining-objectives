# Replication package for *What Do Small Overtrained Language Models Reveal About Deployment Objectives?*

This package reproduces every empirical result in the accompanying anonymous
paper: the within-generation parameter--token estimate, the released-allocation
figure, and the Llama 3 loss-surface calibration and robustness checks.

## Quick start

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run_all.py
python verify.py
```

`requirements-lock.txt` records the environment used for verification;
`requirements.txt` gives broader compatible ranges. The analysis is offline
after installing the public Python dependencies.
It reads only the two CSV files under `data/` and writes all estimates to
`results/` and the paper figure to `figures/`.

## Package map

| Paper result | Command | Generated output |
|---|---|---|
| Within-generation fixed-effects estimate and clustered standard error | `analysis/panel_analysis.py` | `results/panel_estimates.json` |
| Generation-level slopes and token-intensity contrasts | `analysis/panel_analysis.py` | `results/within_generation_estimates.csv` |
| Released-allocation figure | `analysis/plot_allocations.py` | `figures/allocation_contrasts.pdf` |
| Constrained and unconstrained Llama 3 surfaces | `analysis/llama_calibration.py` | `results/llama_calibration.json` |
| Calibration table | `analysis/llama_calibration.py` | `results/llama_calibration_table.csv` |
| Local-slope and leave-one-budget-out checks | `analysis/llama_calibration.py` | `results/local_rate_sensitivity.csv`, `results/leave_one_budget_out.csv` |

`run_all.py` executes these steps in order. `verify.py` checks the generated
files against all numerical values quoted in the paper. No manuscript source,
author information, Git history, private data, credentials, or absolute local
paths are included.

## Statistical conventions

The panel regression absorbs generation fixed effects. Its clustered standard
error uses the finite-sample correction

```text
G/(G-1) * (n-1)/(n-G-1),
```

where `G=10` generations and `n=31` models. The Llama analysis fits

```text
loss = E + A * N^(-alpha) + B * D^(-beta)
```

by nonlinear least squares. The constrained fit imposes Meta's reported token
frontier exponent and flagship allocation; the alternative estimates all five
parameters from the public IsoFLOP observations. The inverse calculation
recovers the total parameter-side wedge

```text
Q(delta) = 3 * [R(N,D) * (N + delta) - D].
```

It does not separately identify expected inference demand from other benefits
or costs of compactness.

## Data provenance and scope

See [`data/README.md`](data/README.md). The model panel contains released
configurations from official papers, model cards, or developer reports. The
IsoFLOP observations are the Llama 3 subset of the public Open Athena table.
Both datasets describe public releases or experiments; neither contains
personal, private, or user-level data.

## Reproducibility

Generated PDFs contain the author metadata `Anonymous`. Numerical outputs are
plain JSON or CSV. The package intentionally excludes cached environments,
compiled files, exploratory analyses, and machine-specific paths.
