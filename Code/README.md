# Research code

This directory contains the Python implementation, numerical checks, tests, configurations, and saved results for the interval Pareto set coverage study. The repository root README gives the current reproduction commands and identifies the Bank Marketing source data.

- `src/interval_pareto/`: interval, objective, Pareto, and regret utilities.
- `experiments/`: synthetic studies, fixed-regularity checks, Bank Marketing preparation, and finite-pool coverage analysis.
- `validation/` and `tests/`: independent numerical checks and automated tests.
- `kaggle/`: notebook builders and the larger experiment runner.
- `configs/` and `kaggle/configs/`: experiment settings.
- `data/bank_marketing/`: the attributed UCI archive, derived stream, and provenance metadata.
- `results/`: saved numerical outputs used in the reports.

The notebooks in `../notebooks/` are the executable Kaggle versions. The Bank finite-pool analysis is described in `EXP12_BANK_PARETO_COVERAGE_REPORT.md`; its saved numerical audit is under `results/real_bank_pareto_coverage/`. Earlier experiment reports remain in this directory for provenance and should be read with their stated assumptions and limitations.
