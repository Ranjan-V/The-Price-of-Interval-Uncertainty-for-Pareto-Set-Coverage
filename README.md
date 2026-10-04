# Interval uncertainty for Pareto set coverage: reproducibility files

This repository contains the code, executable notebooks, inputs, and numerical results for the study *The Price of Interval Uncertainty for Pareto Set Coverage*. The manuscript is not stored here.

## Contents

- [`Code/src/`](Code/src/), [`Code/experiments/`](Code/experiments/), and [`Code/kaggle/`](Code/kaggle/): implementation and experiment runners.
- [`Code/tests/`](Code/tests/) and [`Code/validation/`](Code/validation/): automated and independent numerical checks.
- [`notebooks/`](notebooks/): the Kaggle notebooks used for synthetic, fixed-regularity, and Bank Marketing runs.
- [`Code/data/bank_marketing/`](Code/data/bank_marketing/): UCI source archive, derived `.npz` stream, and source/preprocessing metadata.
- [`Code/results/`](Code/results/): numerical outputs and figures. The Bank finite-pool run is in [`real_bank_pareto_coverage/`](Code/results/real_bank_pareto_coverage/).
- [`Code/EXP12_BANK_PARETO_COVERAGE_REPORT.md`](Code/EXP12_BANK_PARETO_COVERAGE_REPORT.md): data split, exact interval construction, metric, and limitations.

## Reproduce the Bank finite-pool analysis

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r Code/requirements.txt
python Code/experiments/real_bank_pareto_coverage.py
python Code/validation/audit_bank_oracle.py
python -m pytest Code/tests
```

The Bank source archive is the UCI [Bank Marketing dataset](https://doi.org/10.24432/C5K306), attributed to S. Moro, P. Rita, and P. Cortez under CC BY 4.0. [`Code/data/bank_marketing/README.md`](Code/data/bank_marketing/README.md) and its metadata JSON record the source and SHA-256 digest. The experiment treats each chronological batch as a fixed finite pool; it does not infer future-population coverage.

The synthetic suite and notebook builders are documented in [`Code/kaggle/README.md`](Code/kaggle/README.md). The saved notebooks preserve the code used on Kaggle; see their individual cells for the corresponding input package and run settings.
